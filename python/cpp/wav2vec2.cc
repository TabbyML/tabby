#include "module.h"

#include <ctranslate2/models/wav2vec2.h>

#include "replica_pool.h"

namespace ctranslate2 {
  namespace python {

    class Wav2Vec2Wrapper : public ReplicaPoolHelper<models::Wav2Vec2> {
    public:
      using ReplicaPoolHelper::ReplicaPoolHelper;

      StorageView encode(const StorageView& features, const bool to_cpu) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        return _pool->encode(features, to_cpu).get();
      }
    };


    void register_wav2vec2(py::module& m) {
      py::class_<Wav2Vec2Wrapper>(
        m, "Wav2Vec2",
        R"pbdoc(
            Implements the Wav2Vec2 speech recognition model published by Facebook.

            See Also:
               https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec
        )pbdoc")

        .def(py::init<const std::string&, const std::string&, const std::variant<int, std::vector<int>>&, const StringOrMap&, size_t, size_t, long, bool, bool, py::object>(),
             py::arg("model_path"),
             py::arg("device")="cpu",
             py::kw_only(),
             py::arg("device_index")=0,
             py::arg("compute_type")="default",
             py::arg("inter_threads")=1,
             py::arg("intra_threads")=0,
             py::arg("max_queued_batches")=0,
             py::arg("flash_attention")=false,
             py::arg("tensor_parallel")=false,
             py::arg("files")=py::none(),
             R"pbdoc(
                 Initializes a Wav2Vec2 model from a converted model.

                 Arguments:
                   model_path: Path to the CTranslate2 model directory.
                   device: Device to use (possible values are: cpu, cuda, auto).
                   device_index: Device IDs where to place this model on.
                   compute_type: Model computation type or a dictionary mapping a device name
                     to the computation type (possible values are: default, auto, int8, int8_float32,
                     int8_float16, int8_bfloat16, int16, float16, bfloat16, float32).
                   inter_threads: Number of workers to allow executing multiple batches in parallel.
                   intra_threads: Number of OpenMP threads per worker (0 to use a default value).
                   max_queued_batches: Maximum numbers of batches in the worker queue (-1 for unlimited,
                     0 for an automatic value). When the queue is full, future requests will block
                     until a free slot is available.
                   flash_attention: run model with flash attention 2 for self-attention layer
                   tensor_parallel: run model with tensor parallel mode
                   files: Load model files from the memory. This argument is a dictionary mapping
                     file names to file contents as file-like or bytes objects. If this is set,
                     :obj:`model_path` acts as an identifier for this model.
             )pbdoc")

        .def_property_readonly("device", &Wav2Vec2Wrapper::device,
                               "Device this model is running on.")
        .def_property_readonly("device_index", &Wav2Vec2Wrapper::device_index,
                               "List of device IDs where this model is running on.")
        .def_property_readonly("compute_type", &Wav2Vec2Wrapper::compute_type,
                               "Computation type used by the model.")
        .def_property_readonly("num_workers", &Wav2Vec2Wrapper::num_replicas,
                               "Number of model workers backing this instance.")
        .def_property_readonly("num_queued_batches", &Wav2Vec2Wrapper::num_queued_batches,
                               "Number of batches waiting to be processed.")
        .def_property_readonly("tensor_parallel", &Wav2Vec2Wrapper::tensor_parallel,
                               "Run model with tensor parallel mode.")
        .def_property_readonly("num_active_batches", &Wav2Vec2Wrapper::num_active_batches,
                               "Number of batches waiting to be processed or currently processed.")

        .def("encode", &Wav2Vec2Wrapper::encode,
             py::arg("features"),
             py::arg("to_cpu")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Encodes the input features.

                 Arguments:
                   features: Mel spectogram of the audio, as a float array with shape
                     ``[batch_size, 80, 3000]``.
                   to_cpu: Copy the encoder output to the CPU before returning the value.

                 Returns:
                   The encoder output.
             )pbdoc")

        .def("unload_model", &Wav2Vec2Wrapper::unload_model,
             py::arg("to_cpu")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Unloads the model attached to this wav2vec2 but keep enough runtime context
                 to quickly resume wav2vec2 on the initial device.

                 Arguments:
                   to_cpu: If ``True``, the model is moved to the CPU memory and not fully unloaded.
             )pbdoc")

        .def("load_model", &Wav2Vec2Wrapper::load_model,
             py::arg("keep_cache")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Loads the model back to the initial device.

                 Arguments:
                   keep_cache: If ``True``, the model cache in the CPU memory is not deleted if it exists.
             )pbdoc")

        .def_property_readonly("model_is_loaded", &Wav2Vec2Wrapper::model_is_loaded,
                               "Whether the model is loaded on the initial device and ready to be used.")
        ;
    }

  }
}
