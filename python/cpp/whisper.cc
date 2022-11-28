#include "module.h"

#include <ctranslate2/models/whisper.h>

#include "storage_view.h"
#include "replica_pool.h"

namespace ctranslate2 {
  namespace python {

    class WhisperWrapper : public ReplicaPoolHelper<models::Whisper> {
    public:
      using ReplicaPoolHelper::ReplicaPoolHelper;

      std::variant<std::vector<GenerationResult>, std::vector<AsyncResult<GenerationResult>>>
      generate(StorageViewWrapper features,
               std::variant<BatchTokens, BatchIds> prompts,
               bool asynchronous,
               size_t beam_size,
               size_t num_hypotheses,
               float length_penalty,
               float repetition_penalty,
               size_t no_repeat_ngram_size,
               size_t max_length,
               bool return_scores,
               size_t sampling_topk,
               float sampling_temperature) {
        std::vector<std::future<GenerationResult>> futures;

        models::WhisperOptions options;
        options.beam_size = beam_size;
        options.length_penalty = length_penalty;
        options.repetition_penalty = repetition_penalty;
        options.no_repeat_ngram_size = no_repeat_ngram_size;
        options.sampling_topk = sampling_topk;
        options.sampling_temperature = sampling_temperature;
        options.max_length = max_length;
        options.num_hypotheses = num_hypotheses;
        options.return_scores = return_scores;

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
      py::class_<WhisperWrapper>(
        m, "Whisper",
        R"pbdoc(
            Implements the Whisper speech recognition model published by OpenAI.

            See Also:
               https://github.com/openai/whisper
        )pbdoc")

        .def(py::init<const std::string&, const std::string&, const std::variant<int, std::vector<int>>&, const StringOrMap&, size_t, size_t, long>(),
             py::arg("model_path"),
             py::arg("device")="cpu",
             py::kw_only(),
             py::arg("device_index")=0,
             py::arg("compute_type")="default",
             py::arg("inter_threads")=1,
             py::arg("intra_threads")=0,
             py::arg("max_queued_batches")=0,
             R"pbdoc(
                 Initializes a Whisper model from a converted model.

                 Arguments:
                   model_path: Path to the CTranslate2 model directory.
                   device: Device to use (possible values are: cpu, cuda, auto).
                   device_index: Device IDs where to place this model on.
                   compute_type: Model computation type or a dictionary mapping a device name
                        to the computation type
                        (possible values are: default, auto, int8, int8_float16, int16, float16, float).
                   inter_threads: Number of workers to allow executing multiple batches in parallel.
                   intra_threads: Number of OpenMP threads per worker (0 to use a default value).
                   max_queued_batches: Maximum numbers of batches in the worker queue (-1 for unlimited,
                     0 for an automatic value). When the queue is full, future requests will block
                     until a free slot is available.
             )pbdoc")

        .def("generate", &WhisperWrapper::generate,
             py::arg("features"),
             py::arg("prompts"),
             py::kw_only(),
             py::arg("asynchronous")=false,
             py::arg("beam_size")=5,
             py::arg("num_hypotheses")=1,
             py::arg("length_penalty")=1,
             py::arg("repetition_penalty")=1,
             py::arg("no_repeat_ngram_size")=0,
             py::arg("max_length")=448,
             py::arg("return_scores")=false,
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
                   num_hypotheses: Number of hypotheses to return (must be <= :obj:`beam_size`).
                   length_penalty: Exponential penalty applied to the length during beam search.
                   repetition_penalty: Penalty applied to the score of previously generated tokens
                     (set > 1 to penalize).
                   no_repeat_ngram_size: Prevent repetitions of ngrams with this size
                     (set 0 to disable).
                   max_length: Maximum generation length.
                   return_scores: Include the scores in the output.
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
             )pbdoc")

        ;
    }

  }
}
