#include <optional>
#include <unordered_map>
#include <variant>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ctranslate2/translator_pool.h>

namespace py = pybind11;

using StringOrMap = std::variant<std::string, std::unordered_map<std::string, std::string>>;

class ComputeTypeResolver {
private:
  const std::string _device;

public:
  ComputeTypeResolver(std::string device)
    : _device(std::move(device)) {
  }

  ctranslate2::ComputeType
  operator()(const std::string& compute_type) const {
    return ctranslate2::str_to_compute_type(compute_type);
  }

  ctranslate2::ComputeType
  operator()(const std::unordered_map<std::string, std::string>& compute_type) const {
    auto it = compute_type.find(_device);
    if (it == compute_type.end())
      return ctranslate2::ComputeType::DEFAULT;
    return operator()(it->second);
  }
};

struct DeviceIndexResolver {
  std::vector<int> operator()(int device_index) const {
    return std::vector<int>(1, device_index);
  }

  std::vector<int> operator()(const std::vector<int>& device_index) const {
    return device_index;
  }
};

using Tokens = std::vector<std::string>;
using BatchTokens = std::vector<Tokens>;
using BatchTokensOptional = std::optional<std::vector<std::optional<Tokens>>>;

static BatchTokens finalize_optional_batch(const BatchTokensOptional& optional) {
  // Convert missing values to empty vectors.
  BatchTokens batch;
  if (!optional)
    return batch;
  batch.reserve(optional->size());
  for (const auto& tokens : *optional) {
    batch.emplace_back(tokens.value_or(Tokens()));
  }
  return batch;
}

static inline std::vector<int>
get_translators_location(const std::vector<ctranslate2::Translator>& translators) {
  std::vector<int> ids;
  ids.reserve(translators.size());
  for (const auto& translator : translators)
    ids.emplace_back(translator.device_index());
  return ids;
}


class TranslatorWrapper
{
public:
  TranslatorWrapper(const std::string& model_path,
                    const std::string& device,
                    const std::variant<int, std::vector<int>>& device_index,
                    const StringOrMap& compute_type,
                    size_t inter_threads,
                    size_t intra_threads)
    : _model_path(model_path)
    , _device(ctranslate2::str_to_device(device))
    , _compute_type(std::visit(ComputeTypeResolver(device), compute_type))
    , _translator_pool(inter_threads,
                       intra_threads,
                       model_path,
                       _device,
                       std::visit(DeviceIndexResolver(), device_index),
                       _compute_type)
    , _device_index(get_translators_location(_translator_pool.get_translators()))
    , _model_is_loaded(true) {
  }

  bool model_is_loaded() const {
    return _model_is_loaded;
  }

  std::string device() const {
    return ctranslate2::device_to_str(_device);
  }

  const std::vector<int>& device_index() const {
    return _device_index;
  }

  size_t num_translators() const {
    return _translator_pool.num_translators();
  }

  size_t num_queued_batches() {
    return _translator_pool.num_queued_batches();
  }

  using TokenizeFn = std::function<std::vector<std::string>(const std::string&)>;
  using DetokenizeFn = std::function<std::string(const std::vector<std::string>&)>;

  py::tuple translate_file(const std::string& source_path,
                           const std::string& output_path,
                           size_t max_batch_size,
                           size_t read_batch_size,
                           const std::string& batch_type,
                           size_t beam_size,
                           size_t num_hypotheses,
                           float length_penalty,
                           float coverage_penalty,
                           size_t max_decoding_length,
                           size_t min_decoding_length,
                           bool use_vmap,
                           bool with_scores,
                           size_t sampling_topk,
                           float sampling_temperature,
                           const TokenizeFn& tokenize_fn,
                           const DetokenizeFn& detokenize_fn,
                           const std::string& target_path,
                           const TokenizeFn& target_tokenize_fn,
                           bool replace_unknowns) {
    if (bool(tokenize_fn) != bool(detokenize_fn))
      throw std::invalid_argument("tokenize_fn and detokenize_fn should both be set or none at all");
    const std::string* target_path_ptr = target_path.empty() ? nullptr : &target_path;
    if (target_path_ptr && tokenize_fn && !target_tokenize_fn)
      throw std::invalid_argument("target_tokenize_fn should be set when passing a target file");

    assert_model_is_ready();
    ctranslate2::TranslationStats stats;

    {
      py::gil_scoped_release release;

      ctranslate2::TranslationOptions options;
      options.max_batch_size = max_batch_size;
      options.batch_type = ctranslate2::str_to_batch_type(batch_type);
      options.beam_size = beam_size;
      options.length_penalty = length_penalty;
      options.coverage_penalty = coverage_penalty;
      options.sampling_topk = sampling_topk;
      options.sampling_temperature = sampling_temperature;
      options.max_decoding_length = max_decoding_length;
      options.min_decoding_length = min_decoding_length;
      options.num_hypotheses = num_hypotheses;
      options.use_vmap = use_vmap;
      options.return_scores = with_scores;
      options.replace_unknowns = replace_unknowns;

      if (read_batch_size == 0)
        read_batch_size = max_batch_size;

      if (tokenize_fn && detokenize_fn) {
        // Re-acquire the GIL before calling the tokenization functions.
        const auto safe_tokenize_fn = [&tokenize_fn](const std::string& text) {
          py::gil_scoped_acquire acquire;
          return tokenize_fn(text);
        };

        const auto safe_target_tokenize_fn = [&target_tokenize_fn](const std::string& text) {
          py::gil_scoped_acquire acquire;
          return target_tokenize_fn(text);
        };

        const auto safe_detokenize_fn = [&detokenize_fn](const std::vector<std::string>& tokens) {
          py::gil_scoped_acquire acquire;
          return detokenize_fn(tokens);
        };

        stats = _translator_pool.consume_raw_text_file(source_path,
                                                       target_path_ptr,
                                                       output_path,
                                                       safe_tokenize_fn,
                                                       safe_target_tokenize_fn,
                                                       safe_detokenize_fn,
                                                       read_batch_size,
                                                       options,
                                                       with_scores);
      } else {
        stats = _translator_pool.consume_text_file(source_path,
                                                   output_path,
                                                   read_batch_size,
                                                   options,
                                                   with_scores,
                                                   target_path_ptr);
      }
    }

    return py::make_tuple(stats.num_tokens, stats.num_examples, stats.total_time_in_ms);
  }

  py::list translate_batch(const BatchTokens& source,
                           const BatchTokensOptional& target_prefix,
                           size_t max_batch_size,
                           const std::string& batch_type,
                           size_t beam_size,
                           size_t num_hypotheses,
                           float length_penalty,
                           float coverage_penalty,
                           size_t max_decoding_length,
                           size_t min_decoding_length,
                           bool use_vmap,
                           bool return_scores,
                           bool return_attention,
                           bool return_alternatives,
                           size_t sampling_topk,
                           float sampling_temperature,
                           bool replace_unknowns) {
    if (source.empty())
      return py::list();

    assert_model_is_ready();

    std::vector<ctranslate2::TranslationResult> results;

    {
      py::gil_scoped_release release;

      ctranslate2::TranslationOptions options;
      options.max_batch_size = max_batch_size;
      options.batch_type = ctranslate2::str_to_batch_type(batch_type);
      options.beam_size = beam_size;
      options.length_penalty = length_penalty;
      options.coverage_penalty = coverage_penalty;
      options.sampling_topk = sampling_topk;
      options.sampling_temperature = sampling_temperature;
      options.max_decoding_length = max_decoding_length;
      options.min_decoding_length = min_decoding_length;
      options.num_hypotheses = num_hypotheses;
      options.use_vmap = use_vmap;
      options.return_scores = return_scores;
      options.return_attention = return_attention;
      options.return_alternatives = return_alternatives;
      options.replace_unknowns = replace_unknowns;

      results = _translator_pool.translate_batch(source,
                                                 finalize_optional_batch(target_prefix),
                                                 options);
    }

    py::list py_results(results.size());
    for (size_t b = 0; b < results.size(); ++b) {
      const auto& result = results[b];
      py::list batch(result.num_hypotheses());
      for (size_t i = 0; i < result.num_hypotheses(); ++i) {
        py::dict hyp;
        hyp["tokens"] = result.hypotheses()[i];
        if (result.has_scores()) {
          hyp["score"] = result.scores()[i];
        }
        if (result.has_attention()) {
          hyp["attention"] = result.attention()[i];
        }
        batch[i] = hyp;
      }
      py_results[b] = batch;
    }

    return py_results;
  }

  void unload_model(const bool to_cpu) {
    if (!_model_is_loaded || (to_cpu && _device == ctranslate2::Device::CPU))
      return;

    py::gil_scoped_release release;

    const auto& translators = _translator_pool.get_translators();
    if (to_cpu)
      _cached_models.reserve(translators.size());

    for (const auto& translator : translators) {
      if (to_cpu) {
        const auto& model = translator.get_model();
        const_cast<ctranslate2::models::Model&>(*model).set_device(ctranslate2::Device::CPU);
        _cached_models.emplace_back(model);
      }

      const_cast<ctranslate2::Translator&>(translator).detach_model();

      // Clear cache of memory allocator associated with this translator.
      auto* allocator = translator.get_allocator();
      if (allocator && _device == ctranslate2::Device::CUDA)
        allocator->clear_cache();
    }

    _model_is_loaded = false;
  }

  void load_model() {
    if (_model_is_loaded)
      return;

    py::gil_scoped_release release;

    if (_cached_models.empty()) {
      _cached_models = ctranslate2::models::load_replicas(_model_path,
                                                          _device,
                                                          _device_index,
                                                          _compute_type);
    }

    const auto& translators = _translator_pool.get_translators();

    for (size_t i = 0; i < _cached_models.size(); ++i) {
      const auto& model = _cached_models[i];
      const auto& translator = translators[i];

      // If the model was unloaded to the system memory, move it back to the initial device.
      if (model->device() != _device)
        const_cast<ctranslate2::models::Model&>(*model).set_device(_device, _device_index[i]);

      // Reattach the model to the translator instance.
      const_cast<ctranslate2::Translator&>(translator).set_model(model);
    }

    _cached_models.clear();
    _model_is_loaded = true;
  }

private:
  const std::string _model_path;
  const ctranslate2::Device _device;
  const ctranslate2::ComputeType _compute_type;

  ctranslate2::TranslatorPool _translator_pool;
  const std::vector<int> _device_index;

  std::vector<std::shared_ptr<const ctranslate2::models::Model>> _cached_models;
  bool _model_is_loaded;

  void assert_model_is_ready() const {
    if (!model_is_loaded())
      throw std::runtime_error("The model for this translator was unloaded");
  }
};

PYBIND11_MODULE(translator, m)
{
  m.def("contains_model", &ctranslate2::models::contains_model, py::arg("path"));

  py::class_<TranslatorWrapper>(m, "Translator")
    .def(py::init<const std::string&, const std::string&, const std::variant<int, std::vector<int>>&, const StringOrMap&, size_t, size_t>(),
         py::arg("model_path"),
         py::arg("device")="cpu",
         py::arg("device_index")=0,
         py::arg("compute_type")="default",
         py::arg("inter_threads")=1,
         py::arg("intra_threads")=4)
    .def_property_readonly("device", &TranslatorWrapper::device)
    .def_property_readonly("device_index", &TranslatorWrapper::device_index)
    .def_property_readonly("num_translators", &TranslatorWrapper::num_translators)
    .def_property_readonly("num_queued_batches", &TranslatorWrapper::num_queued_batches)
    .def("translate_batch", &TranslatorWrapper::translate_batch,
         py::arg("source"),
         py::arg("target_prefix")=py::none(),
         py::arg("max_batch_size")=0,
         py::arg("batch_type")="examples",
         py::arg("beam_size")=2,
         py::arg("num_hypotheses")=1,
         py::arg("length_penalty")=0,
         py::arg("coverage_penalty")=0,
         py::arg("max_decoding_length")=250,
         py::arg("min_decoding_length")=1,
         py::arg("use_vmap")=false,
         py::arg("return_scores")=true,
         py::arg("return_attention")=false,
         py::arg("return_alternatives")=false,
         py::arg("sampling_topk")=1,
         py::arg("sampling_temperature")=1,
         py::arg("replace_unknowns")=false)
    .def("translate_file", &TranslatorWrapper::translate_file,
         py::arg("input_path"),
         py::arg("output_path"),
         py::arg("max_batch_size")=32,
         py::arg("read_batch_size")=0,
         py::arg("batch_type")="examples",
         py::arg("beam_size")=2,
         py::arg("num_hypotheses")=1,
         py::arg("length_penalty")=0,
         py::arg("coverage_penalty")=0,
         py::arg("max_decoding_length")=250,
         py::arg("min_decoding_length")=1,
         py::arg("use_vmap")=false,
         py::arg("with_scores")=false,
         py::arg("sampling_topk")=1,
         py::arg("sampling_temperature")=1,
         py::arg("tokenize_fn")=nullptr,
         py::arg("detokenize_fn")=nullptr,
         py::arg("target_path")="",
         py::arg("target_tokenize_fn")=nullptr,
         py::arg("replace_unknowns")=false)
    .def("unload_model", &TranslatorWrapper::unload_model,
         py::arg("to_cpu")=false)
    .def("load_model", &TranslatorWrapper::load_model)
    .def_property_readonly("model_is_loaded", &TranslatorWrapper::model_is_loaded)
    ;
}
