#include "module.h"

#include <ctranslate2/models/whisper.h>

#include "storage_view.h"
#include "replica_pool.h"

namespace ctranslate2 {
  namespace python {

    class WhisperWrapper : public ReplicaPoolHelper<models::Whisper> {
    public:
      using ReplicaPoolHelper::ReplicaPoolHelper;

      bool is_multilingual() const {
        return _pool->is_multilingual();
      }

      std::variant<std::vector<models::WhisperGenerationResult>,
                   std::vector<AsyncResult<models::WhisperGenerationResult>>>
      generate(StorageViewWrapper features,
               std::variant<BatchTokens, BatchIds> prompts,
               bool asynchronous,
               size_t beam_size,
               float patience,
               size_t num_hypotheses,
               float length_penalty,
               float repetition_penalty,
               size_t no_repeat_ngram_size,
               size_t max_length,
               bool return_scores,
               bool return_no_speech_prob,
               size_t sampling_topk,
               float sampling_temperature) {
        std::vector<std::future<models::WhisperGenerationResult>> futures;

        models::WhisperOptions options;
        options.beam_size = beam_size;
        options.patience = patience;
        options.length_penalty = length_penalty;
        options.repetition_penalty = repetition_penalty;
        options.no_repeat_ngram_size = no_repeat_ngram_size;
        options.sampling_topk = sampling_topk;
        options.sampling_temperature = sampling_temperature;
        options.max_length = max_length;
        options.num_hypotheses = num_hypotheses;
        options.return_scores = return_scores;
        options.return_no_speech_prob = return_no_speech_prob;

        if (prompts.index() == 0)
          futures = _pool->generate(features.get_view(), std::get<BatchTokens>(prompts), options);
        else
          futures = _pool->generate(features.get_view(), std::get<BatchIds>(prompts), options);

        return maybe_wait_on_futures(std::move(futures), asynchronous);
      }

      std::vector<std::vector<std::pair<std::string, float>>>
      detect_language(StorageViewWrapper features) {
        auto futures = _pool->detect_language(features.get_view());

        std::vector<std::vector<std::pair<std::string, float>>> results;
        results.reserve(futures.size());
        for (auto& future : futures)
          results.emplace_back(future.get());
        return results;
      }
    };


    void register_whisper(py::module& m) {
      py::class_<models::WhisperGenerationResult>(m, "WhisperGenerationResult",
                                                  "A generation result from the Whisper model.")

        .def_readonly("sequences", &models::WhisperGenerationResult::sequences,
                      "Generated sequences of tokens.")
        .def_readonly("sequences_ids", &models::WhisperGenerationResult::sequences_ids,
                      "Generated sequences of token IDs.")
        .def_readonly("scores", &models::WhisperGenerationResult::scores,
                      "Score of each sequence (empty if :obj:`return_scores` was disabled).")
        .def_readonly("no_speech_prob", &models::WhisperGenerationResult::no_speech_prob,
                      "Probability of the no speech token (0 if :obj:`return_no_speech_prob` was disabled).")

        .def("__repr__", [](const models::WhisperGenerationResult& result) {
          return "WhisperGenerationResult(sequences=" + std::string(py::repr(py::cast(result.sequences)))
            + ", sequences_ids=" + std::string(py::repr(py::cast(result.sequences_ids)))
            + ", scores=" + std::string(py::repr(py::cast(result.scores)))
            + ", no_speech_prob=" + std::string(py::repr(py::cast(result.no_speech_prob)))
            + ")";
        })
        ;

      declare_async_wrapper<models::WhisperGenerationResult>(m, "WhisperGenerationResultAsync");

      py::class_<WhisperWrapper>(
        m, "Whisper",
        R"pbdoc(
            Implements the Whisper speech recognition model published by OpenAI.

            See Also:
               https://github.com/openai/whisper
        )pbdoc")

        .def_property_readonly("is_multilingual", &WhisperWrapper::is_multilingual,
                               "Returns ``True`` if this model is multilingual.")

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
                 Initializes a Whisper model from a converted model.

                 Arguments:
                   model_path: Path to the CTranslate2 model directory.
                   device: Device to use (possible values are: cpu, cuda, auto).
                   device_index: Device IDs where to place this model on.
                   compute_type: Model computation type or a dictionary mapping a device name
                        to the computation type
                        (possible values are: default, auto, int8, int8_float16, int16, float16, float32).
                   inter_threads: Number of workers to allow executing multiple batches in parallel.
                   intra_threads: Number of OpenMP threads per worker (0 to use a default value).
                   max_queued_batches: Maximum numbers of batches in the worker queue (-1 for unlimited,
                     0 for an automatic value). When the queue is full, future requests will block
                     until a free slot is available.
                   files: Load model files from the memory. This argument is a dictionary mapping
                     file names to file contents as file-like or bytes objects. If this is set,
                     :obj:`model_path` acts as an identifier for this model.
             )pbdoc")

        .def("generate", &WhisperWrapper::generate,
             py::arg("features"),
             py::arg("prompts"),
             py::kw_only(),
             py::arg("asynchronous")=false,
             py::arg("beam_size")=5,
             py::arg("patience")=1,
             py::arg("num_hypotheses")=1,
             py::arg("length_penalty")=1,
             py::arg("repetition_penalty")=1,
             py::arg("no_repeat_ngram_size")=0,
             py::arg("max_length")=448,
             py::arg("return_scores")=false,
             py::arg("return_no_speech_prob")=false,
             py::arg("sampling_topk")=1,
             py::arg("sampling_temperature")=1,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Encodes the input features and generates from the given prompt.

                 Arguments:
                   features: Mel spectogram of the audio, as a float32 array with shape
                     ``[batch_size, 80, 3000]``.
                   prompts: Batch of initial string tokens or token IDs.
                   asynchronous: Run the model asynchronously.
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
                   max_length: Maximum generation length.
                   return_scores: Include the scores in the output.
                   return_no_speech_prob: Include the probability of the no speech token in the
                     result.
                   sampling_topk: Randomly sample predictions from the top K candidates.
                   sampling_temperature: Sampling temperature to generate more random samples.

                 Returns:
                   A list of generation results.
             )pbdoc")

        .def("detect_language", &WhisperWrapper::detect_language,
             py::arg("features"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Returns the probability of each language.

                 Arguments:
                   features: Mel spectogram of the audio, as a float32 array with shape
                     ``[batch_size, 80, 3000]``.

                 Returns:
                   For each batch, a list of pairs (language, probability) ordered from
                   best to worst probability.

                 Raises:
                   RuntimeError: if the model is not multilingual.
             )pbdoc")

        ;
    }

  }
}
