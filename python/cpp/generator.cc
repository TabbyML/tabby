#include "module.h"

#include <ctranslate2/generator.h>

#include "storage_view.h"
#include "replica_pool.h"

namespace ctranslate2 {
  namespace python {

    class GeneratorWrapper : public ReplicaPoolHelper<Generator> {
    public:
      using ReplicaPoolHelper::ReplicaPoolHelper;

      std::variant<std::vector<GenerationResult>,
                   std::vector<AsyncResult<GenerationResult>>>
      generate_batch(const BatchTokens& tokens,
                     size_t max_batch_size,
                     const std::string& batch_type_str,
                     bool asynchronous,
                     size_t beam_size,
                     float patience,
                     size_t num_hypotheses,
                     float length_penalty,
                     float repetition_penalty,
                     size_t no_repeat_ngram_size,
                     bool disable_unk,
                     const std::optional<std::vector<std::vector<std::string>>>& suppress_sequences,
                     const std::optional<std::string>& end_token,
                     size_t max_length,
                     size_t min_length,
                     bool return_scores,
                     bool return_alternatives,
                     float min_alternative_expansion_prob,
                     size_t sampling_topk,
                     float sampling_temperature) {
        if (tokens.empty())
          return {};

        BatchType batch_type = str_to_batch_type(batch_type_str);
        GenerationOptions options;
        options.beam_size = beam_size;
        options.patience = patience;
        options.length_penalty = length_penalty;
        options.repetition_penalty = repetition_penalty;
        options.no_repeat_ngram_size = no_repeat_ngram_size;
        options.disable_unk = disable_unk;
        options.sampling_topk = sampling_topk;
        options.sampling_temperature = sampling_temperature;
        options.max_length = max_length;
        options.min_length = min_length;
        options.num_hypotheses = num_hypotheses;
        options.return_scores = return_scores;
        options.return_alternatives = return_alternatives;
        options.min_alternative_expansion_prob = min_alternative_expansion_prob;
        if (suppress_sequences)
          options.suppress_sequences = suppress_sequences.value();
        if (end_token)
          options.end_token = end_token.value();

        auto futures = _pool->generate_batch_async(tokens, options, max_batch_size, batch_type);
        return maybe_wait_on_futures(std::move(futures), asynchronous);
      }

      std::variant<std::vector<ScoringResult>,
                   std::vector<AsyncResult<ScoringResult>>>
      score_batch(const BatchTokens& tokens,
                  size_t max_batch_size,
                  const std::string& batch_type_str,
                  size_t max_input_length,
                  bool asynchronous) {
        const auto batch_type = str_to_batch_type(batch_type_str);
        ScoringOptions options;
        options.max_input_length = max_input_length;

        auto futures = _pool->score_batch_async(tokens, options, max_batch_size, batch_type);
        return maybe_wait_on_futures(std::move(futures), asynchronous);
      }

      StorageViewWrapper
      forward_batch(const std::variant<BatchTokens, BatchIds, StorageViewWrapper>& inputs,
                    const std::optional<StorageViewWrapper>& lengths,
                    const bool return_log_probs) {
        std::future<StorageView> future;

        switch (inputs.index()) {
        case 0:
          future = _pool->forward_batch_async(std::get<BatchTokens>(inputs), return_log_probs);
          break;
        case 1:
          future = _pool->forward_batch_async(std::get<BatchIds>(inputs), return_log_probs);
          break;
        case 2:
          if (!lengths)
            throw std::invalid_argument("lengths vector is required when passing a dense input");
          const StorageView& ids_view = std::get<StorageViewWrapper>(inputs).get_view();
          const StorageView& lengths_view = lengths.value().get_view();
          future = _pool->forward_batch_async(ids_view, lengths_view, return_log_probs);
          break;
        }

        return StorageViewWrapper(future.get());
      }
    };


    void register_generator(py::module& m) {
      py::class_<GeneratorWrapper>(
        m, "Generator",
        R"pbdoc(
            A text generator.

            Example:

                >>> generator = ctranslate2.Generator("model/", device="cpu")
                >>> generator.generate_batch([["<s>"]], max_length=50, sampling_topk=20)
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
                 Initializes the generator.

                 Arguments:
                   model_path: Path to the CTranslate2 model directory.
                   device: Device to use (possible values are: cpu, cuda, auto).
                   device_index: Device IDs where to place this generator on.
                   compute_type: Model computation type or a dictionary mapping a device name
                     to the computation type
                     (possible values are: default, auto, int8, int8_float16, int16, float16, float32).
                   inter_threads: Maximum number of parallel generations.
                   intra_threads: Number of OpenMP threads per generator (0 to use a default value).
                   max_queued_batches: Maximum numbers of batches in the queue (-1 for unlimited,
                     0 for an automatic value). When the queue is full, future requests will block
                     until a free slot is available.
                   files: Load model files from the memory. This argument is a dictionary mapping
                     file names to file contents as file-like or bytes objects. If this is set,
                     :obj:`model_path` acts as an identifier for this model.
             )pbdoc")

        .def_property_readonly("device", &GeneratorWrapper::device,
                               "Device this generator is running on.")
        .def_property_readonly("device_index", &GeneratorWrapper::device_index,
                               "List of device IDs where this generator is running on.")
        .def_property_readonly("num_generators", &GeneratorWrapper::num_replicas,
                               "Number of generators backing this instance.")
        .def_property_readonly("num_queued_batches", &GeneratorWrapper::num_queued_batches,
                               "Number of batches waiting to be processed.")
        .def_property_readonly("num_active_batches", &GeneratorWrapper::num_active_batches,
                               "Number of batches waiting to be processed or currently processed.")

        .def("generate_batch", &GeneratorWrapper::generate_batch,
             py::arg("start_tokens"),
             py::kw_only(),
             py::arg("max_batch_size")=0,
             py::arg("batch_type")="examples",
             py::arg("asynchronous")=false,
             py::arg("beam_size")=1,
             py::arg("patience")=1,
             py::arg("num_hypotheses")=1,
             py::arg("length_penalty")=1,
             py::arg("repetition_penalty")=1,
             py::arg("no_repeat_ngram_size")=0,
             py::arg("disable_unk")=false,
             py::arg("suppress_sequences")=py::none(),
             py::arg("end_token")=py::none(),
             py::arg("max_length")=512,
             py::arg("min_length")=0,
             py::arg("return_scores")=false,
             py::arg("return_alternatives")=false,
             py::arg("min_alternative_expansion_prob")=0,
             py::arg("sampling_topk")=1,
             py::arg("sampling_temperature")=1,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Generates from a batch of start tokens.

                 Arguments:
                   start_tokens: Batch of start tokens. If the decoder starts from a special
                     start token like ``<s>``, this token should be added to this input.
                   max_batch_size: The maximum batch size. If the number of inputs is greater than
                     :obj:`max_batch_size`, the inputs are sorted by length and split by chunks of
                     :obj:`max_batch_size` examples so that the number of padding positions is
                     minimized.
                   batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
                   asynchronous: Run the generation asynchronously.
                   beam_size: Beam size (1 for greedy search).
                   patience: Beam search patience factor, as described in
                     https://arxiv.org/abs/2204.05424. The decoding will continue until
                     beam_size*patience hypotheses are finished.
                   num_hypotheses: Number of hypotheses to return.
                   length_penalty: Exponential penalty applied to the length during beam search.
                   repetition_penalty: Penalty applied to the score of previously generated tokens
                     (set > 1 to penalize).
                   no_repeat_ngram_size: Prevent repetitions of ngrams with this size
                     (set 0 to disable).
                   disable_unk: Disable the generation of the unknown token.
                   suppress_sequences: Disable the generation of some sequences of tokens.
                   end_token: Stop the decoding on this token (defaults to the model EOS token).
                   max_length: Maximum generation length.
                   min_length: Minimum generation length.
                   return_scores: Include the scores in the output.
                   return_alternatives: Return alternatives at the first unconstrained decoding position.
                   min_alternative_expansion_prob: Minimum initial probability to expand an alternative.
                   sampling_topk: Randomly sample predictions from the top K candidates.
                   sampling_temperature: Sampling temperature to generate more random samples.

                 Returns:
                   A list of generation results.

                 See Also:
                   `GenerationOptions <https://github.com/OpenNMT/CTranslate2/blob/master/include/ctranslate2/generation.h>`_ structure in the C++ library.
             )pbdoc")

        .def("score_batch", &GeneratorWrapper::score_batch,
             py::arg("tokens"),
             py::kw_only(),
             py::arg("max_batch_size")=0,
             py::arg("batch_type")="examples",
             py::arg("max_input_length")=1024,
             py::arg("asynchronous")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Scores a batch of tokens.

                 Arguments:
                   tokens: Batch of tokens to score. If the model expects special start or end tokens,
                     they should also be added to this input.
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

        .def("forward_batch", &GeneratorWrapper::forward_batch,
             py::arg("inputs"),
             py::arg("lengths")=py::none(),
             py::kw_only(),
             py::arg("return_log_probs")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Forwards a batch of sequences in the generator.

                 Arguments:
                   inputs: A batch of sequences either as string tokens or token IDs.
                     This argument can also be a dense int32 array with shape
                     ``[batch_size, max_length]`` (e.g. created from a Numpy array or PyTorch tensor).
                   lengths: The length of each sequence as a int32 array with shape
                     ``[batch_size]``. Required when :obj:`inputs` is a dense array.
                   return_log_probs: If ``True``, the method returns the log probabilties instead
                     of the unscaled logits.

                 Returns:
                   The output logits, or the output log probabilities if :obj:`return_log_probs`
                   is enabled.
             )pbdoc")
        ;
    }

  }
}
