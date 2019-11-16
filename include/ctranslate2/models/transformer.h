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
      size_t current_spec_revision() const override;
      std::unique_ptr<layers::Encoder> make_encoder() const override;
      std::unique_ptr<layers::Decoder> make_decoder() const override;
    protected:
      void register_variable(const std::string& name, StorageView& variable) override;
      void finalize() override;

      size_t _num_heads;
    };

    class TransformerBaseModel : public TransformerModel {
    public:
      TransformerBaseModel(const std::string& path, size_t spec_revision);
    };

    class TransformerBigModel : public TransformerModel {
    public:
      TransformerBigModel(const std::string& path, size_t spec_revision);
    };

    class PositionEncoder
    {
    public:
      PositionEncoder();
      PositionEncoder(const TransformerModel& model, const std::string& scope);
      void operator()(StorageView& input, size_t index = 0);
    private:
      const StorageView& get_position_encoding(size_t max_time, size_t depth, Device device);
      const StorageView* _model_encoding;
      std::unique_ptr<StorageView> _generated_encoding;
    };

    class TransformerFeedForward
    {
    public:
      TransformerFeedForward(const TransformerModel& model, const std::string& scope);
      void operator()(const StorageView& input, StorageView& output);
    private:
      layers::LayerNorm _layer_norm;
      layers::Dense _ff1;
      layers::Dense _ff2;
    };

    class TransformerEncoderLayer
    {
    public:
      TransformerEncoderLayer(const TransformerModel& model, const std::string& scope);
      void operator()(const StorageView& input,
                      const StorageView& lengths,
                      StorageView& output);
    private:
      layers::MultiHeadAttention _self_attention;
      TransformerFeedForward _ff;
    };

    class TransformerDecoderLayer
    {
    public:
      TransformerDecoderLayer(const TransformerModel& model, const std::string& scope);
      void operator()(const StorageView& input,
                      const StorageView& memory,
                      const StorageView& memory_lengths,
                      StorageView& cached_self_attn_keys,
                      StorageView& cached_self_attn_values,
                      StorageView& cached_attn_keys,
                      StorageView& cached_attn_values,
                      StorageView& output,
                      StorageView* attention = nullptr);
    private:
      layers::MultiHeadAttention _self_attention;
      layers::MultiHeadAttention _encoder_attention;
      TransformerFeedForward _ff;
    };

    class TransformerEncoder : public layers::Encoder
    {
    public:
      TransformerEncoder(const TransformerModel& model, const std::string& scope);
      void operator()(const StorageView& ids,
                      const StorageView& lengths,
                      StorageView& output) override;
    private:
      layers::Embeddings _embeddings;
      PositionEncoder _position_encoder;
      layers::LayerNorm _output_norm;
      std::vector<TransformerEncoderLayer> _layers;
    };

    class TransformerDecoder : public layers::Decoder
    {
    public:
      TransformerDecoder(const TransformerModel& model, const std::string& scope);
      void reduce_vocab(const StorageView& ids) override;
      layers::DecoderState initial_state() const override;
      void operator()(size_t step,
                      const StorageView& ids,
                      const StorageView& memory,
                      const StorageView& memory_lengths,
                      layers::DecoderState& state,
                      StorageView* logits = nullptr,
                      StorageView* attention = nullptr) override;
    protected:
      bool should_reorder_state(const std::string& name) const override;
    private:
      layers::Embeddings _embeddings;
      PositionEncoder _position_encoder;
      layers::LayerNorm _output_norm;
      std::vector<TransformerDecoderLayer> _layers;
      layers::Dense _proj;
    };

  }
}
