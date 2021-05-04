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
                         const bool pre_norm = true);

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
      const Activation _activation;
      const Dense _ff1;
      const Dense _ff2;
    };

    class TransformerEncoderLayer : public Layer
    {
    public:
      TransformerEncoderLayer(const models::Model& model,
                              const std::string& scope,
                              const size_t num_heads,
                              const bool pre_norm = true);

      void operator()(const StorageView& input,
                      const StorageView& lengths,
                      StorageView& output,
                      const Padder* padder = nullptr) const;

      DataType output_type() const override {
        return _ff.output_type();
      }

      dim_t output_size() const override {
        return _ff.output_size();
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
                              const size_t num_heads,
                              const bool with_encoder_attention = true,
                              const bool pre_norm = true);

      void operator()(const StorageView& input,
                      const StorageView* memory,
                      const StorageView* memory_lengths,
                      StorageView& cached_self_attn_keys,
                      StorageView& cached_self_attn_values,
                      StorageView* cached_attn_keys,
                      StorageView* cached_attn_values,
                      StorageView& output,
                      StorageView* attention = nullptr,
                      const Padder* padder = nullptr) const;

      DataType output_type() const override {
        return _ff.output_type();
      }

      dim_t output_size() const override {
        return _ff.output_size();
      }

    private:
      const MultiHeadAttention _self_attention;
      const std::unique_ptr<const MultiHeadAttention> _encoder_attention;
      const FeedForwardNetwork _ff;
    };

    class TransformerEncoder : public Encoder
    {
    public:
      TransformerEncoder(const models::Model& model,
                         const std::string& scope,
                         const size_t num_heads,
                         const bool with_position_encoding = true,
                         const bool pre_norm = true);

      void operator()(const StorageView& ids,
                      const StorageView& lengths,
                      StorageView& output) override;

      DataType output_type() const {
        return _layers.back()->output_type();
      }

      dim_t output_size() const {
        return _layers.back()->output_size();
      }

    private:
      const Embeddings _embeddings;
      const ComputeType _compute_type;
      const std::unique_ptr<PositionEncoder> _position_encoder;
      const std::unique_ptr<LayerNorm> _output_norm;
      std::vector<std::unique_ptr<const TransformerEncoderLayer>> _layers;
    };

    class TransformerDecoder : public Decoder
    {
    public:
      TransformerDecoder(const models::Model& model,
                         const std::string& scope,
                         const size_t num_heads,
                         const bool with_position_encoding = true,
                         const bool with_encoder_attention = true,
                         const bool pre_norm = true);

      void set_vocabulary_mask(const StorageView& ids) override;
      void reset_vocabulary_mask() override;
      DecoderState initial_state() const override;

      void operator()(dim_t step,
                      const StorageView& ids,
                      DecoderState& state,
                      StorageView* logits = nullptr,
                      StorageView* attention = nullptr) override;

      DataType output_type() const {
        return _proj.output_type();
      }

      dim_t output_size() const {
        return _proj.output_size();
      }

    protected:
      bool should_reorder_state(const std::string& name) const override;

    private:
      const bool _with_encoder_attention;
      const ComputeType _compute_type;
      const Embeddings _embeddings;
      const std::unique_ptr<PositionEncoder> _position_encoder;
      const std::unique_ptr<LayerNorm> _output_norm;
      std::vector<std::unique_ptr<const TransformerDecoderLayer>> _layers;
      Dense _proj;
    };

  }
}
