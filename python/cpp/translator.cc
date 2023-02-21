#include "module.h"

#include <shared_mutex>

#include <ctranslate2/translator.h>

#include "replica_pool.h"

namespace ctranslate2 {
  namespace python {

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

    class TranslatorWrapper : public ReplicaPoolHelper<Translator>
    {
    public:
      TranslatorWrapper(const std::string& model_path,
                        const std::string& device,
                        const std::variant<int, std::vector<int>>& device_index,
                        const StringOrMap& compute_type,
                        size_t inter_threads,
                        size_t intra_threads,
                        long max_queued_batches,
                        py::object files)
        : ReplicaPoolHelper(model_path,
                            device,
                            device_index,
                            compute_type,
                            inter_threads,
                            intra_threads,
                            max_queued_batches,
                            files)
        , _device(_model_loader.device)
        , _device_index(_model_loader.device_indices)
        , _num_replicas_per_device(_model_loader.num_replicas_per_device)
        , _model_is_loaded(true) {
      }

      bool model_is_loaded() {
        std::shared_lock lock(_mutex);
        return _model_is_loaded;
      }

      using TokenizeFn = std::function<std::vector<std::string>(const std::string&)>;
      using DetokenizeFn = std::function<std::string(const std::vector<std::string>&)>;

      ExecutionStats
      translate_file(const std::string& source_path,
                     const std::string& output_path,
                     const std::optional<std::string>& target_path,
                     size_t max_batch_size,
                     size_t read_batch_size,
                     const std::string& batch_type_str,
                     size_t beam_size,
                     float patience,
                     size_t num_hypotheses,
                     float length_penalty,
                     float coverage_penalty,
                     float repetition_penalty,
                     size_t no_repeat_ngram_size,
                     bool disable_unk,
                     const std::optional<std::vector<std::vector<std::string>>>& suppress_sequences,
                     const std::optional<std::string>& end_token,
                     float prefix_bias_beta,
                     size_t max_input_length,
                     size_t max_decoding_length,
                     size_t min_decoding_length,
                     bool use_vmap,
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

        BatchType batch_type = str_to_batch_type(batch_type_str);
        TranslationOptions options;
        options.beam_size = beam_size;
        options.patience = patience;
        options.length_penalty = length_penalty;
        options.coverage_penalty = coverage_penalty;
        options.repetition_penalty = repetition_penalty;
        options.no_repeat_ngram_size = no_repeat_ngram_size;
        options.disable_unk = disable_unk;
        options.prefix_bias_beta = prefix_bias_beta;
        options.sampling_topk = sampling_topk;
        options.sampling_temperature = sampling_temperature;
        options.max_input_length = max_input_length;
        options.max_decoding_length = max_decoding_length;
        options.min_decoding_length = min_decoding_length;
        options.num_hypotheses = num_hypotheses;
        options.use_vmap = use_vmap;
        options.return_scores = with_scores;
        options.replace_unknowns = replace_unknowns;
        if (suppress_sequences)
          options.suppress_sequences = suppress_sequences.value();
        if (end_token)
          options.end_token = end_token.value();

        std::shared_lock lock(_mutex);
        assert_model_is_ready();

        if (source_tokenize_fn && target_detokenize_fn) {
          return _pool->translate_raw_text_file(source_path,
                                                target_path_ptr,
                                                output_path,
                                                source_tokenize_fn,
                                                target_tokenize_fn,
                                                target_detokenize_fn,
                                                options,
                                                max_batch_size,
                                                read_batch_size,
                                                batch_type,
                                                with_scores);
        } else {
          return _pool->translate_text_file(source_path,
                                            output_path,
                                            options,
                                            max_batch_size,
                                            read_batch_size,
                                            batch_type,
                                            with_scores,
                                            target_path_ptr);
        }
      }

      std::variant<std::vector<TranslationResult>,
                   std::vector<AsyncResult<TranslationResult>>>
      translate_batch(const BatchTokens& source,
                      const BatchTokensOptional& target_prefix,
                      size_t max_batch_size,
                      const std::string& batch_type_str,
                      bool asynchronous,
                      size_t beam_size,
                      float patience,
                      size_t num_hypotheses,
                      float length_penalty,
                      float coverage_penalty,
                      float repetition_penalty,
                      size_t no_repeat_ngram_size,
                      bool disable_unk,
                      const std::optional<std::vector<std::vector<std::string>>>& suppress_sequences,
                      const std::optional<std::string>& end_token,
                      float prefix_bias_beta,
                      size_t max_input_length,
                      size_t max_decoding_length,
                      size_t min_decoding_length,
                      bool use_vmap,
                      bool return_scores,
                      bool return_attention,
                      bool return_alternatives,
                      float min_alternative_expansion_prob,
                      size_t sampling_topk,
                      float sampling_temperature,
                      bool replace_unknowns) {
        if (source.empty())
          return {};

        BatchType batch_type = str_to_batch_type(batch_type_str);
        TranslationOptions options;
        options.beam_size = beam_size;
        options.patience = patience;
        options.length_penalty = length_penalty;
        options.coverage_penalty = coverage_penalty;
        options.repetition_penalty = repetition_penalty;
        options.no_repeat_ngram_size = no_repeat_ngram_size;
        options.disable_unk = disable_unk;
        options.prefix_bias_beta = prefix_bias_beta;
        options.sampling_topk = sampling_topk;
        options.sampling_temperature = sampling_temperature;
        options.max_input_length = max_input_length;
        options.max_decoding_length = max_decoding_length;
        options.min_decoding_length = min_decoding_length;
        options.num_hypotheses = num_hypotheses;
        options.use_vmap = use_vmap;
        options.return_scores = return_scores;
        options.return_attention = return_attention;
        options.return_alternatives = return_alternatives;
        options.min_alternative_expansion_prob = min_alternative_expansion_prob;
        options.replace_unknowns = replace_unknowns;
        if (suppress_sequences)
          options.suppress_sequences = suppress_sequences.value();
        if (end_token)
          options.end_token = end_token.value();

        std::shared_lock lock(_mutex);
        assert_model_is_ready();

        auto futures = _pool->translate_batch_async(source,
                                                    finalize_optional_batch(target_prefix),
                                                    options,
                                                    max_batch_size,
                                                    batch_type);

        return maybe_wait_on_futures(std::move(futures), asynchronous);
      }

      std::variant<std::vector<ScoringResult>,
                   std::vector<AsyncResult<ScoringResult>>>
      score_batch(const BatchTokens& source,
                  const BatchTokens& target,
                  size_t max_batch_size,
                  const std::string& batch_type_str,
                  size_t max_input_length,
                  bool asynchronous) {
        const auto batch_type = str_to_batch_type(batch_type_str);
        ScoringOptions options;
        options.max_input_length = max_input_length;

        std::shared_lock lock(_mutex);
        assert_model_is_ready();

        auto futures = _pool->score_batch_async(source,
                                                target,
                                                options,
                                                max_batch_size,
                                                batch_type);

        return maybe_wait_on_futures(std::move(futures), asynchronous);
      }

      ExecutionStats score_file(const std::string& source_path,
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

        const auto batch_type = str_to_batch_type(batch_type_str);
        ScoringOptions options;
        options.max_input_length = max_input_length;

        std::shared_lock lock(_mutex);
        assert_model_is_ready();

        if (source_tokenize_fn) {
          return _pool->score_raw_text_file(source_path,
                                            target_path,
                                            output_path,
                                            source_tokenize_fn,
                                            target_tokenize_fn,
                                            target_detokenize_fn,
                                            options,
                                            max_batch_size,
                                            read_batch_size,
                                            batch_type,
                                            with_tokens_score);
        } else {
          return _pool->score_text_file(source_path,
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
        if (to_cpu && _device == Device::CPU)
          return;

        // Do not unload the model if some batches are still being processed.
        if (_pool->num_active_batches() > 0)
          return;

        // If the lock is not acquired immediately it means the model is being used
        // in another thread and we can't unload it at this time.
        std::unique_lock lock(_mutex, std::try_to_lock);
        if (!lock || !_model_is_loaded)
          return;

        _cached_models = _pool->detach_models();
        if (to_cpu)
          move_cached_models(Device::CPU, std::vector<int>(_cached_models.size(), 0));
        else
          _cached_models.clear();

        // We clear the CUDA allocator cache to further reduce the memory after unloading the model.
        if (_device == Device::CUDA)
          _pool->clear_cache();

        _model_is_loaded = false;
      }

      void load_model() {
        std::unique_lock lock(_mutex);
        if (_model_is_loaded)
          return;

        if (_cached_models.empty()) {
          _cached_models = _model_loader.load();
        } else {
          move_cached_models(_device, _device_index, _num_replicas_per_device);
        }

        _pool->set_models(_cached_models);
        _cached_models.clear();
        _model_is_loaded = true;
      }

    private:
      const Device _device;
      const std::vector<int>& _device_index;
      const size_t _num_replicas_per_device;

      std::vector<std::shared_ptr<const models::Model>> _cached_models;
      bool _model_is_loaded;

      // Use a shared mutex to protect the model state (loaded/unloaded).
      // Multiple threads can read the model at the same time, but a single thread can change
      // the model state (e.g. load or unload the model).
      std::shared_mutex _mutex;

      void assert_model_is_ready() const {
        if (!_model_is_loaded)
          throw std::runtime_error("The model for this translator was unloaded");
      }

      void move_cached_models(Device device,
                              const std::vector<int>& device_index,
                              size_t num_models_per_device = 1) {
        for (size_t i = 0; i < _cached_models.size(); ++i) {
          auto& model = const_cast<models::Model&>(*_cached_models[i]);
          model.set_device(device, device_index[i / num_models_per_device]);
        }
      }
    };


    void register_translator(py::module& m) {
      py::class_<TranslatorWrapper>(
        m, "Translator",
        R"pbdoc(
            A text translator.

            Example:

                >>> translator = ctranslate2.Translator("model/", device="cpu")
                >>> translator.translate_batch([["▁Hello", "▁world", "!"]])
        )pbdoc")

        .def(py::init<const std::string&, const std::string&, const std::variant<int, std::vector<int>>&, const StringOrMap&, size_t, size_t, long, py::object>(),
             py::arg("model_path"),
             py::arg("device")="cpu",
             py::kw_only(),
             py::arg("device_index")=0,
             py::arg("compute_type")="default",
             py::arg("inter_threads")=1,
             py::arg("intra_threads")=0,
             py::arg("max_queued_batches")=0,
             py::arg("files")=py::none(),
             R"pbdoc(
                 Initializes the translator.

                 Arguments:
                   model_path: Path to the CTranslate2 model directory.
                   device: Device to use (possible values are: cpu, cuda, auto).
                   device_index: Device IDs where to place this generator on.
                   compute_type: Model computation type or a dictionary mapping a device name
                     to the computation type
                     (possible values are: default, auto, int8, int8_float16, int16, float16, float32).
                   inter_threads: Maximum number of parallel translations.
                   intra_threads: Number of OpenMP threads per translator (0 to use a default value).
                   max_queued_batches: Maximum numbers of batches in the queue (-1 for unlimited,
                     0 for an automatic value). When the queue is full, future requests will block
                     until a free slot is available.
                   files: Load model files from the memory. This argument is a dictionary mapping
                     file names to file contents as file-like or bytes objects. If this is set,
                     :obj:`model_path` acts as an identifier for this model.
             )pbdoc")

        .def_property_readonly("device", &TranslatorWrapper::device,
                               "Device this translator is running on.")
        .def_property_readonly("device_index", &TranslatorWrapper::device_index,
                               "List of device IDs where this translator is running on.")
        .def_property_readonly("num_translators", &TranslatorWrapper::num_replicas,
                               "Number of translators backing this instance.")
        .def_property_readonly("num_queued_batches", &TranslatorWrapper::num_queued_batches,
                               "Number of batches waiting to be processed.")
        .def_property_readonly("num_active_batches", &TranslatorWrapper::num_active_batches,
                               "Number of batches waiting to be processed or currently processed.")

        .def("translate_batch", &TranslatorWrapper::translate_batch,
             py::arg("source"),
             py::arg("target_prefix")=py::none(),
             py::kw_only(),
             py::arg("max_batch_size")=0,
             py::arg("batch_type")="examples",
             py::arg("asynchronous")=false,
             py::arg("beam_size")=2,
             py::arg("patience")=1,
             py::arg("num_hypotheses")=1,
             py::arg("length_penalty")=1,
             py::arg("coverage_penalty")=0,
             py::arg("repetition_penalty")=1,
             py::arg("no_repeat_ngram_size")=0,
             py::arg("disable_unk")=false,
             py::arg("suppress_sequences")=py::none(),
             py::arg("end_token")=py::none(),
             py::arg("prefix_bias_beta")=0,
             py::arg("max_input_length")=1024,
             py::arg("max_decoding_length")=256,
             py::arg("min_decoding_length")=1,
             py::arg("use_vmap")=false,
             py::arg("return_scores")=false,
             py::arg("return_attention")=false,
             py::arg("return_alternatives")=false,
             py::arg("min_alternative_expansion_prob")=0,
             py::arg("sampling_topk")=1,
             py::arg("sampling_temperature")=1,
             py::arg("replace_unknowns")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Translates a batch of tokens.

                 Arguments:
                   source: Batch of source tokens.
                   target_prefix: Optional batch of target prefix tokens.
                   max_batch_size: The maximum batch size. If the number of inputs is greater than
                     :obj:`max_batch_size`, the inputs are sorted by length and split by chunks of
                     :obj:`max_batch_size` examples so that the number of padding positions is
                     minimized.
                   batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
                   asynchronous: Run the translation asynchronously.
                   beam_size: Beam size (1 for greedy search).
                   patience: Beam search patience factor, as described in
                     https://arxiv.org/abs/2204.05424. The decoding will continue until
                     beam_size*patience hypotheses are finished.
                   num_hypotheses: Number of hypotheses to return.
                   length_penalty: Exponential penalty applied to the length during beam search.
                   coverage_penalty: Coverage penalty weight applied during beam search.
                   repetition_penalty: Penalty applied to the score of previously generated tokens
                     (set > 1 to penalize).
                   no_repeat_ngram_size: Prevent repetitions of ngrams with this size
                     (set 0 to disable).
                   disable_unk: Disable the generation of the unknown token.
                   suppress_sequences: Disable the generation of some sequences of tokens.
                   end_token: Stop the decoding on this token (defaults to the model EOS token).
                   prefix_bias_beta: Parameter for biasing translations towards given prefix.
                   max_input_length: Truncate inputs after this many tokens (set 0 to disable).
                   max_decoding_length: Maximum prediction length.
                   min_decoding_length: Minimum prediction length.
                   use_vmap: Use the vocabulary mapping file saved in this model
                   return_scores: Include the scores in the output.
                   return_attention: Include the attention vectors in the output.
                   return_alternatives: Return alternatives at the first unconstrained decoding position.
                   min_alternative_expansion_prob: Minimum initial probability to expand an alternative.
                   sampling_topk: Randomly sample predictions from the top K candidates.
                   sampling_temperature: Sampling temperature to generate more random samples.
                   replace_unknowns: Replace unknown target tokens by the source token with the highest attention.

                 Returns:
                   A list of translation results.

                 See Also:
                   `TranslationOptions <https://github.com/OpenNMT/CTranslate2/blob/master/include/ctranslate2/translation.h>`_ structure in the C++ library.
             )pbdoc")

        .def("translate_file", &TranslatorWrapper::translate_file,
             py::arg("source_path"),
             py::arg("output_path"),
             py::arg("target_path")=py::none(),
             py::kw_only(),
             py::arg("max_batch_size")=32,
             py::arg("read_batch_size")=0,
             py::arg("batch_type")="examples",
             py::arg("beam_size")=2,
             py::arg("patience")=1,
             py::arg("num_hypotheses")=1,
             py::arg("length_penalty")=1,
             py::arg("coverage_penalty")=0,
             py::arg("repetition_penalty")=1,
             py::arg("no_repeat_ngram_size")=0,
             py::arg("disable_unk")=false,
             py::arg("suppress_sequences")=py::none(),
             py::arg("end_token")=py::none(),
             py::arg("prefix_bias_beta")=0,
             py::arg("max_input_length")=1024,
             py::arg("max_decoding_length")=256,
             py::arg("min_decoding_length")=1,
             py::arg("use_vmap")=false,
             py::arg("with_scores")=false,
             py::arg("sampling_topk")=1,
             py::arg("sampling_temperature")=1,
             py::arg("replace_unknowns")=false,
             py::arg("source_tokenize_fn")=nullptr,
             py::arg("target_tokenize_fn")=nullptr,
             py::arg("target_detokenize_fn")=nullptr,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Translates a file.

                 Arguments:
                   source_path: Path to the source file.
                   output_path: Path to the output file.
                   target_path: Path to the target prefix file.
                   max_batch_size: The maximum batch size.
                   read_batch_size: The number of examples to read from the file before sorting
                     by length and splitting by chunks of :obj:`max_batch_size` examples
                     (set 0 for an automatic value).
                   batch_type: Whether :obj:`max_batch_size` and :obj:`read_batch_size` are the
                     numbers of "examples" or "tokens".
                   asynchronous: Run the translation asynchronously.
                   beam_size: Beam size (1 for greedy search).
                   patience: Beam search patience factor, as described in
                     https://arxiv.org/abs/2204.05424. The decoding will continue until
                     beam_size*patience hypotheses are finished.
                   num_hypotheses: Number of hypotheses to return.
                   length_penalty: Exponential penalty applied to the length during beam search.
                   coverage_penalty: Coverage penalty weight applied during beam search.
                   repetition_penalty: Penalty applied to the score of previously generated tokens
                     (set > 1 to penalize).
                   no_repeat_ngram_size: Prevent repetitions of ngrams with this size
                     (set 0 to disable).
                   disable_unk: Disable the generation of the unknown token.
                   suppress_sequences: Disable the generation of some sequences of tokens.
                   end_token: Stop the decoding on this token (defaults to the model EOS token).
                   prefix_bias_beta: Parameter for biasing translations towards given prefix.
                   max_input_length: Truncate inputs after this many tokens (set 0 to disable).
                   max_decoding_length: Maximum prediction length.
                   min_decoding_length: Minimum prediction length.
                   use_vmap: Use the vocabulary mapping file saved in this model
                   with_scores: Include the scores in the output.
                   sampling_topk: Randomly sample predictions from the top K candidates.
                   sampling_temperature: Sampling temperature to generate more random samples.
                   replace_unknowns: Replace unknown target tokens by the source token with the highest attention.
                   source_tokenize_fn: Function to tokenize source lines.
                   target_tokenize_fn: Function to tokenize target lines.
                   target_detokenize_fn: Function to detokenize target outputs.

                 Returns:
                   A statistics object.

                 See Also:
                   `TranslationOptions <https://github.com/OpenNMT/CTranslate2/blob/master/include/ctranslate2/translation.h>`_ structure in the C++ library.
             )pbdoc")

        .def("score_batch", &TranslatorWrapper::score_batch,
             py::arg("source"),
             py::arg("target"),
             py::kw_only(),
             py::arg("max_batch_size")=0,
             py::arg("batch_type")="examples",
             py::arg("max_input_length")=1024,
             py::arg("asynchronous")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Scores a batch of parallel tokens.

                 Arguments:
                   source: Batch of source tokens.
                   target: Batch of target tokens.
                   max_batch_size: The maximum batch size. If the number of inputs is greater than
                     :obj:`max_batch_size`, the inputs are sorted by length and split by chunks of
                     :obj:`max_batch_size` examples so that the number of padding positions is
                     minimized.
                   batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
                   max_input_length: Truncate inputs after this many tokens (0 to disable).
                   asynchronous: Run the scoring asynchronously.

                 Returns:
                   A list of scoring results.
             )pbdoc")

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
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Scores a parallel file.

                 Each line in :obj:`output_path` will have the format:

                 .. code-block:: text

                     <score> ||| <target> [||| <score_token_0> <score_token_1> ...]

                 The score is normalized by the target length which includes the end of sentence
                 token ``</s>``.

                 Arguments:
                   source_path: Path to the source file.
                   target_path: Path to the target file.
                   output_path: Path to the output file.
                   max_batch_size: The maximum batch size.
                   read_batch_size: The number of examples to read from the file before sorting
                     by length and splitting by chunks of :obj:`max_batch_size` examples
                     (set 0 for an automatic value).
                   batch_type: Whether :obj:`max_batch_size` and :obj:`read_batch_size` are the
                     number of "examples" or "tokens".
                   max_input_length: Truncate inputs after this many tokens (0 to disable).
                   with_tokens_score: Include the token-level scores in the output file.
                   source_tokenize_fn: Function to tokenize source lines.
                   target_tokenize_fn: Function to tokenize target lines.
                   target_detokenize_fn: Function to detokenize target outputs.

                 Returns:
                   A statistics object.
             )pbdoc")

        .def("unload_model", &TranslatorWrapper::unload_model,
             py::arg("to_cpu")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Unloads the model attached to this translator but keep enough runtime context
                 to quickly resume translation on the initial device. The model is not guaranteed
                 to be unloaded if translations are running concurrently.

                 Arguments:
                   to_cpu: If ``True``, the model is moved to the CPU memory and not fully unloaded.
             )pbdoc")

        .def("load_model", &TranslatorWrapper::load_model,
             py::call_guard<py::gil_scoped_release>(),
             "Loads the model back to the initial device.")

        .def_property_readonly("model_is_loaded", &TranslatorWrapper::model_is_loaded,
                               "Whether the model is loaded on the initial device and ready to be used.")
        ;
    }

  }
}
