#include "ctranslate2/models/wav2vec2.h"

#include <algorithm>

#include "ctranslate2/decoding.h"

#include "dispatch.h"
#include "dtw.h"

#ifdef CT2_WITH_CUDA
#  include "cuda/utils.h"
#endif


namespace ctranslate2 {
  namespace models {

    const Vocabulary& Wav2Vec2Model::get_vocabulary() const {
      return *_vocabulary;
    }

    size_t Wav2Vec2Model::current_spec_revision() const {
      return 3;
    }

    void Wav2Vec2Model::initialize(ModelReader& model_reader) {
      VocabularyInfo vocab_info;
      vocab_info.unk_token = "[UNK]";
      vocab_info.bos_token = "<s>";
      vocab_info.eos_token = "</s>";

      _vocabulary = load_vocabulary(model_reader, "vocabulary", std::move(vocab_info));
      if (!_vocabulary)
        throw std::runtime_error("Cannot load the vocabulary from the model directory");
    }

    bool Wav2Vec2Model::is_quantizable(const std::string& variable_name) const {
      return (Model::is_quantizable(variable_name)
              && variable_name.find("conv") == std::string::npos);
    }

    bool Wav2Vec2Model::is_linear_weight(const std::string& variable_name) const {
      return is_quantizable(variable_name) && variable_name.find("embeddings") == std::string::npos;
    }

    std::unique_ptr<Model> Wav2Vec2Model::clone() const {
      return std::make_unique<Wav2Vec2Model>(*this);
    }


    std::unique_ptr<Wav2Vec2Replica> Wav2Vec2Replica::create_from_model(const Model& model) {
      if (!dynamic_cast<const Wav2Vec2Model*>(&model))
        throw std::invalid_argument("The model is not a Wav2Vec2 model");

      const auto scoped_device_setter = model.get_scoped_device_setter();
      const auto model_ptr = model.shared_from_this();
      const auto concrete_model = std::static_pointer_cast<const Wav2Vec2Model>(model_ptr);
      return std::make_unique<Wav2Vec2Replica>(concrete_model);
    }

    Wav2Vec2Replica::Wav2Vec2Replica(const std::shared_ptr<const Wav2Vec2Model>& model)
      : ModelReplica(model)
      , _model(model)
      , _encoder(std::make_unique<layers::Wav2Vec2Encoder>(*model, "encoder"))
    {
    }


    StorageView Wav2Vec2Replica::encode(StorageView features, const bool to_cpu) {
      PROFILE("Wav2Vec2Replica::encode");

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const Device device = _model->device();
      const DataType dtype = _encoder->output_type();
      features.move_to(device, dtype);

      StorageView encoder_output(dtype, device);
      (*_encoder)(features, encoder_output);

      if (to_cpu) {
        if (device != Device::CPU)
          encoder_output = encoder_output.to(Device::CPU);

        return encoder_output;
      }

      // Ensure all operations are finished before returning the output.
      synchronize_stream(device);

      return encoder_output;
    }

    StorageView Wav2Vec2Replica::maybe_encode(StorageView features) {
      const Device device = _model->device();
      const DataType dtype = _encoder->output_type();

      features.move_to(device, dtype);

      if (_encoder->is_encoded(features))
        return features;

      StorageView encoder_output(dtype, device);
      (*_encoder)(features, encoder_output);
      return encoder_output;
    }

    std::future<StorageView> Wav2Vec2::encode(const StorageView& features, const bool to_cpu) {
      return post<StorageView>(
        [features = features.sync_copy(), to_cpu](Wav2Vec2Replica& replica) mutable {
          return replica.encode(std::move(features), to_cpu);
        });
    }

  }
}
