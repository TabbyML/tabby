#pragma once

#include "ctranslate2/layers/transformer.h"

namespace ctranslate2 {
  namespace layers {

    class Wav2Vec2Encoder : public Layer {
    public:
      Wav2Vec2Encoder(const models::Model& model, const std::string& scope);

      void operator()(const StorageView& features, StorageView& output);

      DataType output_type() const override {
        return _output_norm.output_type();
      }

      dim_t output_size() const override {
        return _output_norm.output_size();
      }

      dim_t input_size() const {
        return 1024;
      }

      bool is_encoded(const StorageView& features) const {
        // Input features shape: [batch_size, input_size, input_time]
        // Encoder output shape: [batch_size, input_time // 2, output_size]
        //
        // input_time is variable so we check that dimension 1 is different than its original value.

        return (features.rank() == 3
                && features.dim(2) == output_size()
                && features.dim(1) != input_size());
      }

    private:
      const ops::GELU _gelu;
      // wav2vec2.encoder modules except pos_conv_embed due to groups=16 being not supported
      //const ops::Transpose _transpose;
      const dim_t _num_heads;
      const std::vector<std::unique_ptr<const TransformerEncoderLayer>> _layers;
      const LayerNorm _output_norm;
    };

  }
}
