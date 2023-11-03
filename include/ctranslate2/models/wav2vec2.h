#pragma once

//#include "ctranslate2/generation.h"
#include "ctranslate2/layers/wav2vec2.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/replica_pool.h"

namespace ctranslate2 {
  namespace models {

    struct Wav2Vec2Options {
      // Maximum generation length.
      size_t max_length = 448;

      // Randomly sample from the top K candidates (set 0 to sample from the full distribution).
      size_t sampling_topk = 1;

      // Maximum index of the first predicted timestamp.
      size_t max_initial_timestamp_index = 50;

      // Suppress blank outputs at the beginning of the sampling.
      bool suppress_blank = true;

      // List of token IDs to suppress.
      // -1 will suppress a default set of symbols as defined in the model config.json file.
      std::vector<int> suppress_tokens = {-1};
    };


    class Wav2Vec2Model : public Model {
    public:
      const Vocabulary& get_vocabulary() const;
      size_t current_spec_revision() const override;
      bool is_quantizable(const std::string& variable_name) const override;
      bool is_linear_weight(const std::string& variable_name) const override;
      std::unique_ptr<Model> clone() const override;

      bool use_global_int16_scale() const override {
        return false;
      }

    protected:
      void initialize(ModelReader& model_reader) override;
    private:
      std::shared_ptr<const Vocabulary> _vocabulary;
    };

    class Wav2Vec2Replica : public ModelReplica {
    public:
      static std::unique_ptr<Wav2Vec2Replica> create_from_model(const Model& model);

      Wav2Vec2Replica(const std::shared_ptr<const Wav2Vec2Model>& model);

      StorageView encode(StorageView features, const bool to_cpu);

    private:
      const std::shared_ptr<const Wav2Vec2Model> _model;
      const std::unique_ptr<layers::Wav2Vec2Encoder> _encoder;

      StorageView maybe_encode(StorageView features);
    };

    class Wav2Vec2 : public ReplicaPool<Wav2Vec2Replica> {
    public:
      using ReplicaPool::ReplicaPool;

      std::future<StorageView> encode(const StorageView& features, const bool to_cpu);

    };

  }
}
