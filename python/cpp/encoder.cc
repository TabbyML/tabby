#include "module.h"

#include <ctranslate2/encoder.h>

#include "replica_pool.h"

namespace ctranslate2 {
  namespace python {

    class EncoderWrapper : public ReplicaPoolHelper<Encoder> {
    public:
      using ReplicaPoolHelper::ReplicaPoolHelper;

      EncoderForwardOutput
      forward_batch(const std::variant<BatchTokens, BatchIds, StorageView>& inputs,
                    const std::optional<StorageView>& lengths,
                    const std::optional<BatchIds>& token_type_ids) {
        std::future<EncoderForwardOutput> future;
        std::shared_lock lock(_mutex);
        assert_model_is_ready();

        switch (inputs.index()) {
        case 0:
          future = _pool->forward_batch_async(
            std::get<BatchTokens>(inputs),
            token_type_ids.value_or(std::vector<std::vector<size_t>>()));
          break;
        case 1:
          future = _pool->forward_batch_async(
            std::get<BatchIds>(inputs),
            token_type_ids.value_or(std::vector<std::vector<size_t>>()));
          break;
        case 2:
          if (!lengths)
            throw std::invalid_argument("lengths vector is required when passing a dense input");
          future = _pool->forward_batch_async(
            std::get<StorageView>(inputs),
            lengths.value(),
            token_type_ids.value_or(std::vector<std::vector<size_t>>()));
          break;
        }

        return future.get();
      }
    };


    void register_encoder(py::module& m) {
      py::class_<EncoderForwardOutput>(m, "EncoderForwardOutput",
                                       "Forward output of an encoder model.")

        .def_readonly("last_hidden_state", &EncoderForwardOutput::last_hidden_state,
                      "Output of the last layer.")
        .def_readonly("pooler_output", &EncoderForwardOutput::pooler_output,
                      "Output of the pooling layer.")

        .def("__repr__", [](const EncoderForwardOutput& output) {
          return "EncoderForwardOutput(last_hidden_state="
            + std::string(py::repr(py::cast(output.last_hidden_state)))
            + ", pooler_output=" + std::string(py::repr(py::cast(output.pooler_output)))
            + ")";
        })
        ;

      py::class_<EncoderWrapper>(
        m, "Encoder",
        R"pbdoc(
            A text encoder.

            Example:

                >>> encoder = ctranslate2.Encoder("model/", device="cpu")
                >>> encoder.forward_batch([["▁Hello", "▁world", "!"]])
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
                 Initializes the encoder.

                 Arguments:
                   model_path: Path to the CTranslate2 model directory.
                   device: Device to use (possible values are: cpu, cuda, auto).
                   device_index: Device IDs where to place this encoder on.
                   compute_type: Model computation type or a dictionary mapping a device name
                     to the computation type (possible values are: default, auto, int8, int8_float32,
                     int8_float16, int8_bfloat16, int16, float16, bfloat16, float32).
                   inter_threads: Maximum number of parallel generations.
                   intra_threads: Number of OpenMP threads per encoder (0 to use a default value).
                   max_queued_batches: Maximum numbers of batches in the queue (-1 for unlimited,
                     0 for an automatic value). When the queue is full, future requests will block
                     until a free slot is available.
                   flash_attention: run model with flash attention 2 for self-attention layer
                   tensor_parallel: run model with tensor parallel mode
                   files: Load model files from the memory. This argument is a dictionary mapping
                     file names to file contents as file-like or bytes objects. If this is set,
                     :obj:`model_path` acts as an identifier for this model.
             )pbdoc")

        .def_property_readonly("device", &EncoderWrapper::device,
                               "Device this encoder is running on.")
        .def_property_readonly("device_index", &EncoderWrapper::device_index,
                               "List of device IDs where this encoder is running on.")
        .def_property_readonly("compute_type", &EncoderWrapper::compute_type,
                               "Computation type used by the model.")
        .def_property_readonly("num_encoders", &EncoderWrapper::num_replicas,
                               "Number of encoders backing this instance.")
        .def_property_readonly("num_queued_batches", &EncoderWrapper::num_queued_batches,
                               "Number of batches waiting to be processed.")
        .def_property_readonly("tensor_parallel", &EncoderWrapper::tensor_parallel,
                               "Run model with tensor parallel mode.")
        .def_property_readonly("num_active_batches", &EncoderWrapper::num_active_batches,
                               "Number of batches waiting to be processed or currently processed.")

        .def("forward_batch", &EncoderWrapper::forward_batch,
             py::arg("inputs"),
             py::arg("lengths")=py::none(),
             py::arg("token_type_ids")=py::none(),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Forwards a batch of sequences in the encoder.

                 Arguments:
                   inputs: A batch of sequences either as string tokens or token IDs.
                     This argument can also be a dense int32 array with shape
                     ``[batch_size, max_length]`` (e.g. created from a Numpy array or PyTorch tensor).
                   lengths: The length of each sequence as a int32 array with shape
                     ``[batch_size]``. Required when :obj:`inputs` is a dense array.
                   token_type_ids: A batch of token type IDs of same shape as :obj:`inputs`.
                     ``[batch_size, max_length]``.

                 Returns:
                   The encoder model output.
             )pbdoc")

        .def("unload_model", &EncoderWrapper::unload_model,
             py::arg("to_cpu")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Unloads the model attached to this encoder but keep enough runtime context
                 to quickly resume encoder on the initial device.

                 Arguments:
                   to_cpu: If ``True``, the model is moved to the CPU memory and not fully unloaded.
             )pbdoc")

        .def("load_model", &EncoderWrapper::load_model,
             py::arg("keep_cache")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Loads the model back to the initial device.

                 Arguments:
                   keep_cache: If ``True``, the model cache in the CPU memory is not deleted if it exists.
             )pbdoc")

        .def_property_readonly("model_is_loaded", &EncoderWrapper::model_is_loaded,
                               "Whether the model is loaded on the initial device and ready to be used.")
        ;
    }

  }
}
