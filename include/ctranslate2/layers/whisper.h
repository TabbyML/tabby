#pragma once

#include "ctranslate2/layers/transformer.h"

namespace ctranslate2 {
  namespace layers {

    class WhisperEncoder : public Layer {
    public:
      WhisperEncoder(const models::Model& model, const std::string& scope);

      void operator()(const StorageView& features, StorageView& output);

      DataType output_type() const override {
        return _output_norm.output_type();
      }

      dim_t output_size() const override {
        return _output_norm.output_size();
      }

    private:
      const Conv1D _conv1;
      const Conv1D _conv2;
      const ops::GELU _gelu;
      const ops::Transpose _transpose;
      PositionEmbedding _position_embedding;
      const dim_t _num_heads;
      const std::vector<std::unique_ptr<const TransformerEncoderLayer>> _layers;
      const LayerNorm _output_norm;
    };

  }
}
