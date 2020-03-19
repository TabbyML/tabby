#pragma once

// This file defines the execution engine for a TransformerSpec model.

#include "ctranslate2/layers/layers.h"

#include "model.h"

namespace ctranslate2 {
  namespace models {

    class TransformerModel : public Model
    {
    public:
      TransformerModel(const std::string& path, size_t spec_revision, size_t num_heads = 0);
      size_t num_heads() const;
      bool with_relative_position() const;
      size_t current_spec_revision() const override;
      std::unique_ptr<layers::Encoder> make_encoder() const override;
      std::unique_ptr<layers::Decoder> make_decoder() const override;
    protected:
      void register_variable(const std::string& name, StorageView& variable) override;
      void finalize() override;

      size_t _num_heads;
      bool _with_relative_position;
    };

    class PositionEncoder
    {
    public:
      PositionEncoder();
      PositionEncoder(const TransformerModel& model, const std::string& scope);
      void operator()(StorageView& input, dim_t index = 0);
    private:
      const StorageView& get_position_encoding(dim_t max_time, dim_t depth, Device device);
      const StorageView* _model_encoding;
      std::unique_ptr<StorageView> _generated_encoding;
    };

    class TransformerFeedForward
    {
    public:
      TransformerFeedForward(const TransformerModel& model, const std::string& scope);
      void operator()(const StorageView& input, StorageView& output) const;
    private:
      const layers::LayerNorm _layer_norm;
      const layers::Dense _ff1;
      const layers::Dense _ff2;
    };

    class TransformerEncoderLayer
    {
    public:
      TransformerEncoderLayer(const TransformerModel& model, const std::string& scope);
      void operator()(const StorageView& input,
                      const StorageView& lengths,
                      StorageView& output) const;
    private:
      const layers::MultiHeadAttention _self_attention;
      const TransformerFeedForward _ff;
    };

    class TransformerDecoderLayer
    {
    public:
      TransformerDecoderLayer(const TransformerModel& model,
                              const std::string& scope,
                              const bool with_encoder_attention = true);
      void operator()(const StorageView& input,
                      const StorageView* memory,
                      const StorageView* memory_lengths,
                      StorageView& cached_self_attn_keys,
                      StorageView& cached_self_attn_values,
                      StorageView* cached_attn_keys,
                      StorageView* cached_attn_values,
                      StorageView& output,
                      StorageView* attention = nullptr) const;
    private:
      const layers::MultiHeadAttention _self_attention;
      const std::unique_ptr<const layers::MultiHeadAttention> _encoder_attention;
      const TransformerFeedForward _ff;
    };

    class TransformerEncoder : public layers::Encoder
    {
    public:
      TransformerEncoder(const TransformerModel& model, const std::string& scope);
      void operator()(const StorageView& ids,
                      const StorageView& lengths,
                      StorageView& output) override;
    private:
      const layers::Embeddings _embeddings;
      const std::unique_ptr<PositionEncoder> _position_encoder;
      const layers::LayerNorm _output_norm;
      std::vector<std::unique_ptr<const TransformerEncoderLayer>> _layers;
    };

    class TransformerDecoder : public layers::Decoder
    {
    public:
      TransformerDecoder(const TransformerModel& model,
                         const std::string& scope,
                         const bool with_encoder_attention = true);
      void set_vocabulary_mask(const StorageView& ids) override;
      void reset_vocabulary_mask() override;
      layers::DecoderState initial_state() const override;
      void operator()(dim_t step,
                      const StorageView& ids,
                      layers::DecoderState& state,
                      StorageView* logits = nullptr,
                      StorageView* attention = nullptr) override;
    protected:
      bool should_reorder_state(const std::string& name) const override;
    private:
      const bool _with_encoder_attention;
      const layers::Embeddings _embeddings;
      const std::unique_ptr<PositionEncoder> _position_encoder;
      const layers::LayerNorm _output_norm;
      std::vector<std::unique_ptr<const TransformerDecoderLayer>> _layers;
      layers::Dense _proj;
    };

  }
}
