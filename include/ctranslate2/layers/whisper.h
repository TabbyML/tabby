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

      dim_t max_output_time() const {
        return _position_embedding.num_positions();
      }

      dim_t input_size() const {
        return _conv1.input_size();
      }

      dim_t max_input_time() const {
        return max_output_time() * 2;
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
      const Conv1D _conv1;
      const Conv1D _conv2;
      const ops::GELU _gelu;
      const ops::Transpose _transpose;
      PositionEmbedding _position_embedding;
      const dim_t _num_heads;
      const std::vector<std::unique_ptr<const TransformerEncoderLayer>> _layers;
      const LayerNorm _output_norm;
    };

    class WhisperDecoder : public TransformerDecoder {
    public:
      using TransformerDecoder::TransformerDecoder;

      bool return_normalized_attention() const override {
        return false;
      }

      void forward_prompt(const StorageView& prompt,
                          DecoderState& state,
                          StorageView* outputs = nullptr);

      void compute_logits_for_steps(const StorageView& outputs,
                                    const StorageView& steps,
                                    StorageView& logits);
    };

  }
}
