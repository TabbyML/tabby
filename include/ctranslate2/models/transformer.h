#pragma once

// This file defines the execution engine for a TransformerSpec model.

#include "ctranslate2/layers/layers.h"

#include "model.h"

namespace ctranslate2 {
  namespace models {

    class TransformerModel : public Model
    {
    public:
      TransformerModel(const std::string& path, size_t spec_revision, size_t num_heads);
      size_t num_heads() const;
      size_t current_spec_revision() const override;
      void register_variable(const std::string& name, StorageView& variable) override;
      std::unique_ptr<Encoder> make_encoder() const override;
      std::unique_ptr<Decoder> make_decoder() const override;
    protected:
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
      PositionEncoder(const TransformerModel& model, const std::string& scope);
      void operator()(StorageView& input, size_t index = 0);
    private:
      const StorageView& get_position_encoding(size_t max_time, size_t depth, Device device) const;
      const StorageView* _encoding;
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

    class DotProductAttention
    {
    public:
      void operator()(const StorageView& queries,
                      const StorageView& keys,
                      const StorageView& values,
                      const StorageView* values_lengths,
                      StorageView& output);
    };

    class MultiHeadAttention
    {
    public:
      MultiHeadAttention(const TransformerModel& model, const std::string& scope);
      void operator()(const StorageView& queries,
                      const StorageView* memory,
                      const StorageView* memory_lengths,
                      StorageView& output,
                      StorageView* cached_keys = nullptr,
                      StorageView* cached_values = nullptr);
    private:
      size_t _num_heads;
      std::vector<layers::Dense> _linear;
      layers::LayerNorm _layer_norm;
      DotProductAttention _attention;
      ops::Transpose _transpose_op;

      void split_heads(const StorageView& x, StorageView& y);
      void combine_heads(const StorageView& x, StorageView& y);
      static void cache_proj(StorageView& proj, StorageView& cache);
    };

    class TransformerEncoderLayer
    {
    public:
      TransformerEncoderLayer(const TransformerModel& model, const std::string& scope);
      void operator()(const StorageView& input,
                      const StorageView& lengths,
                      StorageView& output);
    private:
      MultiHeadAttention _self_attention;
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
                      StorageView& output);
    private:
      MultiHeadAttention _self_attention;
      MultiHeadAttention _encoder_attention;
      TransformerFeedForward _ff;
    };

    class TransformerEncoder : public Encoder
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

    class TransformerDecoder : public Decoder
    {
    public:
      TransformerDecoder(const TransformerModel& model, const std::string& scope);
      DecoderState initial_state() const override;
      void operator()(size_t step,
                      const StorageView& ids,
                      const StorageView& candidates,
                      const StorageView& memory,
                      const StorageView& memory_lengths,
                      DecoderState& state,
                      StorageView& logits) override;
    private:
      layers::Embeddings _embeddings;
      PositionEncoder _position_encoder;
      layers::LayerNorm _output_norm;
      std::vector<TransformerDecoderLayer> _layers;
      layers::Dense _proj;
    };

  }
}
