#include <mutex>
#include <optional>
#include <shared_mutex>
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
    return {device_index};
  }

  std::vector<int> operator()(const std::vector<int>& device_index) const {
    return device_index;
  }
};

using Tokens = std::vector<std::string>;
using BatchTokens = std::vector<Tokens>;
using BatchTokensOptional = std::optional<std::vector<std::optional<Tokens>>>;

// This wrapper re-acquires the GIL before calling a Python function.
template <typename Function>
class SafeCaller {
public:
  SafeCaller(const Function& function)
    : _function(function)
  {
  }

  typename Function::result_type operator()(typename Function::argument_type input) const {
    py::gil_scoped_acquire acquire;
    return _function(input);
  }

private:
  const Function& _function;
};

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
get_translators_location(const ctranslate2::TranslatorPool& translator_pool) {
  const size_t num_translators = translator_pool.num_translators();
  std::vector<int> ids;
  ids.reserve(num_translators);
  for (size_t i = 0; i < num_translators; ++i)
    ids.emplace_back(translator_pool.get_translator(i).device_index());
  return ids;
}


template <typename T>
class AsyncResult {
public:
  AsyncResult(std::future<T> future)
    : _future(std::move(future))
  {
  }

  const T& result() {
    if (!_done) {
      {
        py::gil_scoped_release release;
        _result = _future.get();
      }
      _done = true;  // Assign done attribute while the GIL is held.
    }
    return _result;
  }

  bool done() {
    constexpr std::chrono::seconds zero_sec(0);
    return _done || _future.wait_for(zero_sec) == std::future_status::ready;
  }

private:
  std::future<T> _future;
  T _result;
  bool _done = false;
};


class TranslatorWrapper
{
public:
  TranslatorWrapper(const std::string& model_path,
                    const std::string& device,
                    const std::variant<int, std::vector<int>>& device_index,
                    const StringOrMap& compute_type,
                    size_t inter_threads,
                    size_t intra_threads,
                    long max_queued_batches)
    : _model_path(model_path)
    , _device(ctranslate2::str_to_device(device))
    , _compute_type(std::visit(ComputeTypeResolver(device), compute_type))
    , _translator_pool(inter_threads,
                       intra_threads,
                       model_path,
                       _device,
                       std::visit(DeviceIndexResolver(), device_index),
                       _compute_type,
                       max_queued_batches)
    , _device_index(get_translators_location(_translator_pool))
    , _model_is_loaded(true) {
  }

  bool model_is_loaded() {
    std::shared_lock lock(_mutex);
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

  size_t num_active_batches() const {
    return _translator_pool.num_active_batches();
  }

  using TokenizeFn = std::function<std::vector<std::string>(const std::string&)>;
  using DetokenizeFn = std::function<std::string(const std::vector<std::string>&)>;

  ctranslate2::TranslationStats
  translate_file(const std::string& source_path,
                 const std::string& output_path,
                 const std::optional<std::string>& target_path,
                 size_t max_batch_size,
                 size_t read_batch_size,
                 const std::string& batch_type_str,
                 size_t beam_size,
                 size_t num_hypotheses,
                 float length_penalty,
                 float coverage_penalty,
                 float repetition_penalty,
                 bool disable_unk,
                 float prefix_bias_beta,
                 bool allow_early_exit,
                 size_t max_input_length,
                 size_t max_decoding_length,
                 size_t min_decoding_length,
                 bool use_vmap,
                 bool normalize_scores,
                 bool with_scores,
                 size_t sampling_topk,
                 float sampling_temperature,
                 bool replace_unknowns,
                 const TokenizeFn& source_tokenize_fn,
                 const TokenizeFn& target_tokenize_fn,
                 const DetokenizeFn& target_detokenize_fn) {
    if (bool(source_tokenize_fn) != bool(target_detokenize_fn))
      throw std::invalid_argument("source_tokenize_fn and target_detokenize_fn should both be set or none at all");
    const std::string* target_path_ptr = target_path ? &target_path.value() : nullptr;
    if (target_path_ptr && source_tokenize_fn && !target_tokenize_fn)
      throw std::invalid_argument("target_tokenize_fn should be set when passing a target file");

    ctranslate2::BatchType batch_type = ctranslate2::str_to_batch_type(batch_type_str);
    ctranslate2::TranslationOptions options;
    options.beam_size = beam_size;
    options.length_penalty = length_penalty;
    options.coverage_penalty = coverage_penalty;
    options.repetition_penalty = repetition_penalty;
    options.disable_unk = disable_unk;
    options.prefix_bias_beta = prefix_bias_beta;
    options.allow_early_exit = allow_early_exit;
    options.sampling_topk = sampling_topk;
    options.sampling_temperature = sampling_temperature;
    options.max_input_length = max_input_length;
    options.max_decoding_length = max_decoding_length;
    options.min_decoding_length = min_decoding_length;
    options.num_hypotheses = num_hypotheses;
    options.use_vmap = use_vmap;
    options.normalize_scores = normalize_scores;
    options.return_scores = with_scores;
    options.replace_unknowns = replace_unknowns;

    std::shared_lock lock(_mutex);
    assert_model_is_ready();

    if (source_tokenize_fn && target_detokenize_fn) {
      const SafeCaller<TokenizeFn> safe_source_tokenize_fn(source_tokenize_fn);
      const SafeCaller<TokenizeFn> safe_target_tokenize_fn(target_tokenize_fn);
      const SafeCaller<DetokenizeFn> safe_target_detokenize_fn(target_detokenize_fn);
      return _translator_pool.consume_raw_text_file(source_path,
                                                    target_path_ptr,
                                                    output_path,
                                                    safe_source_tokenize_fn,
                                                    safe_target_tokenize_fn,
                                                    safe_target_detokenize_fn,
                                                    options,
                                                    max_batch_size,
                                                    read_batch_size,
                                                    batch_type,
                                                    with_scores);
    } else {
      return _translator_pool.consume_text_file(source_path,
                                                output_path,
                                                options,
                                                max_batch_size,
                                                read_batch_size,
                                                batch_type,
                                                with_scores,
                                                target_path_ptr);
    }
  }

  std::variant<std::vector<ctranslate2::TranslationResult>,
               std::vector<AsyncResult<ctranslate2::TranslationResult>>>
  translate_batch(const BatchTokens& source,
                  const BatchTokensOptional& target_prefix,
                  size_t max_batch_size,
                  const std::string& batch_type_str,
                  bool asynchronous,
                  size_t beam_size,
                  size_t num_hypotheses,
                  float length_penalty,
                  float coverage_penalty,
                  float repetition_penalty,
                  bool disable_unk,
                  float prefix_bias_beta,
                  bool allow_early_exit,
                  size_t max_input_length,
                  size_t max_decoding_length,
                  size_t min_decoding_length,
                  bool use_vmap,
                  bool normalize_scores,
                  bool return_scores,
                  bool return_attention,
                  bool return_alternatives,
                  size_t sampling_topk,
                  float sampling_temperature,
                  bool replace_unknowns) {
    if (source.empty())
      return {};

    ctranslate2::BatchType batch_type = ctranslate2::str_to_batch_type(batch_type_str);
    ctranslate2::TranslationOptions options;
    options.beam_size = beam_size;
    options.length_penalty = length_penalty;
    options.coverage_penalty = coverage_penalty;
    options.repetition_penalty = repetition_penalty;
    options.disable_unk = disable_unk;
    options.prefix_bias_beta = prefix_bias_beta;
    options.allow_early_exit = allow_early_exit;
    options.sampling_topk = sampling_topk;
    options.sampling_temperature = sampling_temperature;
    options.max_input_length = max_input_length;
    options.max_decoding_length = max_decoding_length;
    options.min_decoding_length = min_decoding_length;
    options.num_hypotheses = num_hypotheses;
    options.use_vmap = use_vmap;
    options.normalize_scores = normalize_scores;
    options.return_scores = return_scores;
    options.return_attention = return_attention;
    options.return_alternatives = return_alternatives;
    options.replace_unknowns = replace_unknowns;

    std::shared_lock lock(_mutex);
    assert_model_is_ready();

    auto futures = _translator_pool.translate_batch_async(source,
                                                          finalize_optional_batch(target_prefix),
                                                          options,
                                                          max_batch_size,
                                                          batch_type);

    if (asynchronous) {
      std::vector<AsyncResult<ctranslate2::TranslationResult>> results;
      results.reserve(futures.size());
      for (auto& future : futures)
        results.emplace_back(std::move(future));
      return std::move(results);
    } else {
      std::vector<ctranslate2::TranslationResult> results;
      results.reserve(futures.size());
      for (auto& future : futures)
        results.emplace_back(future.get());
      return std::move(results);
    }
  }

  std::vector<std::vector<float>>
  score_batch(const BatchTokens& source,
              const BatchTokens& target,
              size_t max_batch_size,
              const std::string& batch_type_str,
              size_t max_input_length) {
    const auto batch_type = ctranslate2::str_to_batch_type(batch_type_str);
    ctranslate2::ScoringOptions options;
    options.max_input_length = max_input_length;

    std::shared_lock lock(_mutex);
    assert_model_is_ready();

    auto results = _translator_pool.score_batch(source, target, options, max_batch_size, batch_type);

    std::vector<std::vector<float>> scores;
    scores.reserve(results.size());
    for (auto& result : results)
      scores.emplace_back(std::move(result.tokens_score));
    return scores;
  }

  ctranslate2::TranslationStats score_file(const std::string& source_path,
                                           const std::string& target_path,
                                           const std::string& output_path,
                                           size_t max_batch_size,
                                           size_t read_batch_size,
                                           const std::string& batch_type_str,
                                           size_t max_input_length,
                                           bool with_tokens_score,
                                           const TokenizeFn& source_tokenize_fn,
                                           const TokenizeFn& target_tokenize_fn,
                                           const DetokenizeFn& target_detokenize_fn) {
    if (bool(source_tokenize_fn) != bool(target_tokenize_fn)
        || bool(target_tokenize_fn) != bool(target_detokenize_fn))
      throw std::invalid_argument("source_tokenize_fn, target_tokenize_fn, and target_detokenize_fn should all be set or none at all");

    const auto batch_type = ctranslate2::str_to_batch_type(batch_type_str);
    ctranslate2::ScoringOptions options;
    options.max_input_length = max_input_length;

    std::shared_lock lock(_mutex);
    assert_model_is_ready();

    if (source_tokenize_fn) {
      const SafeCaller<TokenizeFn> safe_source_tokenize_fn(source_tokenize_fn);
      const SafeCaller<TokenizeFn> safe_target_tokenize_fn(target_tokenize_fn);
      const SafeCaller<DetokenizeFn> safe_target_detokenize_fn(target_detokenize_fn);
      return _translator_pool.score_raw_text_file(source_path,
                                                  target_path,
                                                  output_path,
                                                  safe_source_tokenize_fn,
                                                  safe_target_tokenize_fn,
                                                  safe_target_detokenize_fn,
                                                  options,
                                                  max_batch_size,
                                                  read_batch_size,
                                                  batch_type,
                                                  with_tokens_score);
    } else {
      return _translator_pool.score_text_file(source_path,
                                              target_path,
                                              output_path,
                                              options,
                                              max_batch_size,
                                              read_batch_size,
                                              batch_type,
                                              with_tokens_score);
    }
  }

  void unload_model(const bool to_cpu) {
    if (to_cpu && _device == ctranslate2::Device::CPU)
      return;

    // Do not unload the model if some batches are still being processed.
    if (_translator_pool.num_active_batches() > 0)
      return;

    // If the lock is not acquired immediately it means the model is being used
    // in another thread and we can't unload it at this time.
    std::unique_lock lock(_mutex, std::try_to_lock);
    if (!lock || !_model_is_loaded)
      return;

    const size_t num_translators = _translator_pool.num_translators();
    if (to_cpu)
      _cached_models.reserve(num_translators);

    for (size_t i = 0; i < num_translators; ++i) {
      const auto& translator = _translator_pool.get_translator(i);

      {
        const auto model = const_cast<ctranslate2::Translator&>(translator).detach_model();

        if (to_cpu) {
          const_cast<ctranslate2::models::Model&>(*model).set_device(ctranslate2::Device::CPU);
          _cached_models.emplace_back(model);
        }
      }
    }

    // We clear the CUDA allocator cache to further reduce the memory after unloading the model.
    if (_device == ctranslate2::Device::CUDA)
      _translator_pool.clear_cache();

    _model_is_loaded = false;
  }

  void load_model() {
    std::unique_lock lock(_mutex);
    if (_model_is_loaded)
      return;

    if (_cached_models.empty()) {
      _cached_models = ctranslate2::models::load_replicas(_model_path,
                                                          _device,
                                                          _device_index,
                                                          _compute_type);
    }

    for (size_t i = 0; i < _cached_models.size(); ++i) {
      const auto& model = _cached_models[i];
      const auto& translator = _translator_pool.get_translator(i);

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

  // Use a shared mutex to protect the model state (loaded/unloaded).
  // Multiple threads can read the model at the same time, but a single thread can change
  // the model state (e.g. load or unload the model).
  std::shared_mutex _mutex;

  void assert_model_is_ready() const {
    if (!_model_is_loaded)
      throw std::runtime_error("The model for this translator was unloaded");
  }
};

static py::set get_supported_compute_types(const std::string& device_str, const int device_index) {
  const auto device = ctranslate2::str_to_device(device_str);

  const bool support_float16 = ctranslate2::mayiuse_float16(device, device_index);
  const bool support_int16 = ctranslate2::mayiuse_int16(device, device_index);
  const bool support_int8 = ctranslate2::mayiuse_int8(device, device_index);

  py::set compute_types;
  compute_types.add("float");
  if (support_float16)
    compute_types.add("float16");
  if (support_int16)
    compute_types.add("int16");
  if (support_int8)
    compute_types.add("int8");
  if (support_int8 && support_float16)
    compute_types.add("int8_float16");
  return compute_types;
}

template <typename T>
static void declare_async_wrapper(py::module& m, const char* name) {
  py::class_<AsyncResult<T>>(m, name)
    .def("result", &AsyncResult<T>::result)
    .def("done", &AsyncResult<T>::done)
    ;
}

PYBIND11_MODULE(translator, m)
{
  m.def("contains_model", &ctranslate2::models::contains_model, py::arg("path"));
  m.def("get_cuda_device_count", &ctranslate2::get_gpu_count);
  m.def("get_supported_compute_types", &get_supported_compute_types,
        py::arg("device"),
        py::arg("device_index")=0);
  m.def("set_random_seed", &ctranslate2::set_random_seed, py::arg("seed"));

  py::class_<ctranslate2::TranslationResult>(m, "TranslationResult")
    .def_readonly("hypotheses", &ctranslate2::TranslationResult::hypotheses)
    .def_readonly("scores", &ctranslate2::TranslationResult::scores)
    .def_readonly("attention", &ctranslate2::TranslationResult::attention)
    .def("__repr__", [](const ctranslate2::TranslationResult& result) {
      return "TranslationResult(hypotheses=" + std::string(py::repr(py::cast(result.hypotheses)))
        + ", scores=" + std::string(py::repr(py::cast(result.scores)))
        + ", attention=" + std::string(py::repr(py::cast(result.attention)))
        + ")";
    })

    // Backward compatibility with using translate_batch output as a list of dicts.
    .def("__len__", &ctranslate2::TranslationResult::num_hypotheses)
    .def("__getitem__", [](const ctranslate2::TranslationResult& result, size_t i) {
      if (i >= result.num_hypotheses())
        throw std::out_of_range("list index out of range");
      py::dict hypothesis;
      hypothesis["tokens"] = result.hypotheses[i];
      if (result.has_scores())
        hypothesis["score"] = result.scores[i];
      if (result.has_attention())
        hypothesis["attention"] = result.attention[i];
      return hypothesis;
    })
    ;

  declare_async_wrapper<ctranslate2::TranslationResult>(m, "AsyncTranslationResult");

  py::class_<TranslatorWrapper>(m, "Translator")
    .def(py::init<const std::string&, const std::string&, const std::variant<int, std::vector<int>>&, const StringOrMap&, size_t, size_t, long>(),
         py::arg("model_path"),
         py::arg("device")="cpu",
         py::kw_only(),
         py::arg("device_index")=0,
         py::arg("compute_type")="default",
         py::arg("inter_threads")=1,
         py::arg("intra_threads")=0,
         py::arg("max_queued_batches")=0)
    .def_property_readonly("device", &TranslatorWrapper::device)
    .def_property_readonly("device_index", &TranslatorWrapper::device_index)
    .def_property_readonly("num_translators", &TranslatorWrapper::num_translators)
    .def_property_readonly("num_queued_batches", &TranslatorWrapper::num_queued_batches)
    .def_property_readonly("num_active_batches", &TranslatorWrapper::num_active_batches)
    .def("translate_batch", &TranslatorWrapper::translate_batch,
         py::arg("source"),
         py::arg("target_prefix")=py::none(),
         py::kw_only(),
         py::arg("max_batch_size")=0,
         py::arg("batch_type")="examples",
         py::arg("asynchronous")=false,
         py::arg("beam_size")=2,
         py::arg("num_hypotheses")=1,
         py::arg("length_penalty")=0,
         py::arg("coverage_penalty")=0,
         py::arg("repetition_penalty")=1,
         py::arg("disable_unk")=false,
         py::arg("prefix_bias_beta")=0,
         py::arg("allow_early_exit")=true,
         py::arg("max_input_length")=1024,
         py::arg("max_decoding_length")=256,
         py::arg("min_decoding_length")=1,
         py::arg("use_vmap")=false,
         py::arg("normalize_scores")=false,
         py::arg("return_scores")=false,
         py::arg("return_attention")=false,
         py::arg("return_alternatives")=false,
         py::arg("sampling_topk")=1,
         py::arg("sampling_temperature")=1,
         py::arg("replace_unknowns")=false,
         py::call_guard<py::gil_scoped_release>())
    .def("translate_file", &TranslatorWrapper::translate_file,
         py::arg("source_path"),
         py::arg("output_path"),
         py::arg("target_path")=py::none(),
         py::kw_only(),
         py::arg("max_batch_size")=32,
         py::arg("read_batch_size")=0,
         py::arg("batch_type")="examples",
         py::arg("beam_size")=2,
         py::arg("num_hypotheses")=1,
         py::arg("length_penalty")=0,
         py::arg("coverage_penalty")=0,
         py::arg("repetition_penalty")=1,
         py::arg("disable_unk")=false,
         py::arg("prefix_bias_beta")=0,
         py::arg("allow_early_exit")=true,
         py::arg("max_input_length")=1024,
         py::arg("max_decoding_length")=256,
         py::arg("min_decoding_length")=1,
         py::arg("use_vmap")=false,
         py::arg("normalize_scores")=false,
         py::arg("with_scores")=false,
         py::arg("sampling_topk")=1,
         py::arg("sampling_temperature")=1,
         py::arg("replace_unknowns")=false,
         py::arg("source_tokenize_fn")=nullptr,
         py::arg("target_tokenize_fn")=nullptr,
         py::arg("target_detokenize_fn")=nullptr,
         py::call_guard<py::gil_scoped_release>())
    .def("score_batch", &TranslatorWrapper::score_batch,
         py::arg("source"),
         py::arg("target"),
         py::kw_only(),
         py::arg("max_batch_size")=0,
         py::arg("batch_type")="examples",
         py::arg("max_input_length")=1024,
         py::call_guard<py::gil_scoped_release>())
    .def("score_file", &TranslatorWrapper::score_file,
         py::arg("source_path"),
         py::arg("target_path"),
         py::arg("output_path"),
         py::kw_only(),
         py::arg("max_batch_size")=32,
         py::arg("read_batch_size")=0,
         py::arg("batch_type")="examples",
         py::arg("max_input_length")=1024,
         py::arg("with_tokens_score")=false,
         py::arg("source_tokenize_fn")=nullptr,
         py::arg("target_tokenize_fn")=nullptr,
         py::arg("target_detokenize_fn")=nullptr,
         py::call_guard<py::gil_scoped_release>())
    .def("unload_model", &TranslatorWrapper::unload_model,
         py::arg("to_cpu")=false,
         py::call_guard<py::gil_scoped_release>())
    .def("load_model", &TranslatorWrapper::load_model,
         py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("model_is_loaded", &TranslatorWrapper::model_is_loaded)
    ;

  py::class_<ctranslate2::TranslationStats>(m, "TranslationStats")
    .def_readonly("num_tokens", &ctranslate2::TranslationStats::num_tokens)
    .def_readonly("num_examples", &ctranslate2::TranslationStats::num_examples)
    .def_readonly("total_time_in_ms", &ctranslate2::TranslationStats::total_time_in_ms)
    .def("__repr__", [](const ctranslate2::TranslationStats& stats) {
      return "TranslationStats(num_tokens=" + std::string(py::repr(py::cast(stats.num_tokens)))
        + ", num_examples=" + std::string(py::repr(py::cast(stats.num_examples)))
        + ", total_time_in_ms=" + std::string(py::repr(py::cast(stats.total_time_in_ms)))
        + ")";
    })

    // Backward compatibility with using translate_file output as a tuple.
    .def("__getitem__", [](const ctranslate2::TranslationStats& stats, size_t index) {
      auto tuple = py::make_tuple(stats.num_tokens, stats.num_examples, stats.total_time_in_ms);
      return py::object(tuple[index]);
    })
    ;
}
