#pragma once

#include "ctranslate2/layers/attention.h"
#include "ctranslate2/layers/common.h"
#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/layers/encoder.h"
#include "ctranslate2/padder.h"

namespace ctranslate2 {
  namespace layers {

    class FeedForwardNetwork : public Layer
    {
    public:
      FeedForwardNetwork(const models::Model& model,
                         const std::string& scope,
                         const bool pre_norm = true,
                         const ops::ActivationType activation_type = ops::ActivationType::ReLU);

      void operator()(const StorageView& input, StorageView& output) const;

      DataType output_type() const override {
        return _ff2.output_type();
      }

      dim_t output_size() const override {
        return _ff2.output_size();
      }

    private:
      const LayerNorm _layer_norm;
      const bool _pre_norm;
      const ops::ActivationType _activation_type;
      const Dense _ff1;
      const std::unique_ptr<const Dense> _ff1_noact;
      const Dense _ff2;
    };

    class TransformerEncoderLayer : public Layer
    {
    public:
      TransformerEncoderLayer(const models::Model& model,
                              const std::string& scope,
                              const dim_t num_heads,
                              const bool pre_norm = true,
                              const ops::ActivationType activation_type = ops::ActivationType::ReLU);

      void operator()(const StorageView& input,
                      const StorageView* lengths,
                      StorageView& output,
                      const Padder* padder = nullptr) const;

      DataType output_type() const override {
        return _ff.output_type();
      }

      dim_t output_size() const override {
        return _ff.output_size();
      }

      bool has_relative_position() const {
        return _self_attention.has_relative_position();
      }

    private:
      const MultiHeadAttention _self_attention;
      const FeedForwardNetwork _ff;
    };

    class TransformerDecoderLayer : public Layer
    {
    public:
      TransformerDecoderLayer(const models::Model& model,
                              const std::string& scope,
                              const dim_t num_heads,
                              const bool pre_norm = true,
                              const ops::ActivationType activation_type = ops::ActivationType::ReLU);

      void operator()(const StorageView& input,
                      const StorageView* input_lengths,
                      const StorageView* memory,
                      const StorageView* memory_lengths,
                      StorageView* cached_self_attn_keys,
                      StorageView* cached_self_attn_values,
                      StorageView* cached_attn_keys,
                      StorageView* cached_attn_values,
                      StorageView& output,
                      StorageView* attention = nullptr,
                      const Padder* input_padder = nullptr,
                      const Padder* memory_padder = nullptr) const;

      DataType output_type() const override {
        return _ff.output_type();
      }

      dim_t output_size() const override {
        return _ff.output_size();
      }

      bool has_cross_attention() const {
        return bool(_encoder_attention);
      }

      bool has_relative_position() const {
        return _self_attention.has_relative_position();
      }

    private:
      const MultiHeadAttention _self_attention;
      const std::unique_ptr<const MultiHeadAttention> _encoder_attention;
      const FeedForwardNetwork _ff;
    };

    class TransformerEncoder : public Encoder
    {
    public:
      TransformerEncoder(const models::Model& model, const std::string& scope);

      void operator()(const std::vector<StorageView>& ids,
                      const StorageView* lengths,
                      StorageView& output) override;

      size_t num_input_features() const override {
        return _embeddings.num_inputs();
      }

      DataType output_type() const override {
        return _layers.back()->output_type();
      }

      dim_t output_size() const override {
        return _layers.back()->output_size();
      }

    private:
      const ParallelEmbeddings _embeddings;
      const std::unique_ptr<const StorageView> _embeddings_scale;
      const dim_t _num_heads;
      const ComputeType _compute_type;
      const std::unique_ptr<const LayerNorm> _layernorm_embedding;
      const std::unique_ptr<const LayerNorm> _output_norm;
      const std::vector<std::unique_ptr<const TransformerEncoderLayer>> _layers;
      const std::unique_ptr<PositionEncoder> _position_encoder;
    };

    class TransformerDecoder : public Decoder
    {
    public:
      TransformerDecoder(const models::Model& model, const std::string& scope);

      DecoderState initial_state(bool iterative_decoding = true) const override;
      bool replicate_state(const std::string& name) const override;

      void operator()(dim_t step,
                      const StorageView& ids,
                      DecoderState& state,
                      StorageView* logits = nullptr,
                      StorageView* attention = nullptr) override;
      void operator()(const StorageView& ids,
                      const StorageView& lengths,
                      DecoderState& state,
                      StorageView& logits) override;

    protected:
      Dense& output_layer() override {
        return _proj;
      }

      void decode(const StorageView& ids,
                  const StorageView* lengths,
                  dim_t step,
                  DecoderState& state,
                  StorageView* outputs = nullptr,
                  StorageView* attention = nullptr,
                  bool return_logits = true);

      const dim_t _num_heads;
      const ComputeType _compute_type;
      const Embeddings _embeddings;
      const bool _start_from_zero_embedding;
      const std::unique_ptr<const StorageView> _embeddings_scale;
      std::unique_ptr<const StorageView> _outputs_scale;
      const std::unique_ptr<const LayerNorm> _layernorm_embedding;
      const std::unique_ptr<const LayerNorm> _output_norm;
      const std::unique_ptr<const Dense> _project_in;
      const std::unique_ptr<const Dense> _project_out;
      const std::vector<std::unique_ptr<const TransformerDecoderLayer>> _layers;
      const std::unique_ptr<PositionEncoder> _position_encoder;
      const bool _with_encoder_attention;
      dim_t _alignment_layer;
      dim_t _alignment_heads;
      Dense _proj;
    };

  }
}
