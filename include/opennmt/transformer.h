#pragma once

#include <map>

#include "decoder.h"
#include "encoder.h"
#include "ops/ops.h"
#include "storage_view.h"

namespace opennmt {

  class Model
  {
  public:
    Model(const std::string& path);
    const StorageView& get_variable(const std::string& scope) const;
  private:
    std::map<std::string, StorageView> _variable_index;
  };

  class ScaledEmbeddings
  {
  public:
    ScaledEmbeddings(const Model& model, const std::string& scope);
    StorageView& operator()(const StorageView& ids);
  private:
    ops::Gather _gather_op;
    const StorageView& _embeddings;
    StorageView _gathered;
    StorageView _output;
  };

  class PositionEncoder
  {
  public:
    StorageView& operator()(const StorageView& input, const StorageView& lengths);
    StorageView& operator()(const StorageView& input, size_t index);
  private:
    size_t _max_cached_time = 500;
    StorageView _cached_encodings;
    StorageView _output;
    void precompute_position_encoding(size_t max_time, size_t depth);
  };

  class Dense
  {
  public:
    Dense(const Model& model, const std::string& scope);
    StorageView& operator()(const StorageView& input);
  private:
    ops::Gemm _gemm_op;
    const StorageView& _weight;
    const StorageView& _bias;
    StorageView _output;
  };

  class LayerNorm
  {
  public:
    LayerNorm(const Model& model, const std::string& scope);
    StorageView& operator()(const StorageView& input);
  private:
    ops::LayerNorm _norm_op;
    const StorageView& _beta;
    const StorageView& _gamma;
    StorageView _output;
  };

  class TransformerFeedForward
  {
  public:
    TransformerFeedForward(const Model& model, const std::string& scope);
    StorageView& operator()(const StorageView& input);
  private:
    LayerNorm _layer_norm;
    Dense _ff1;
    Dense _ff2;
  };

  class DotProductAttention
  {
  public:
    StorageView& operator()(const StorageView& queries,
                            const StorageView& keys,
                            const StorageView& values,
                            const StorageView& values_lengths);
  private:
    StorageView _dot;
    StorageView _attn;
  };

  class MultiHeadAttention
  {
  public:
    MultiHeadAttention(const Model& model, const std::string& scope, size_t num_heads);
    StorageView& compute_attention(const StorageView& queries,
                                   const StorageView& keys,
                                   const StorageView& values,
                                   const StorageView& queries_lengths,
                                   const StorageView& values_lengths);
  protected:
    LayerNorm _layer_norm;
  private:
    size_t _num_heads;
    DotProductAttention _attention;
    ops::Transpose _transpose_op;
    StorageView _split_queries;
    StorageView _split_keys;
    StorageView _split_values;
    StorageView _combined;
    void split_heads(const StorageView& x, StorageView& y);
    void combine_heads(const StorageView& x, StorageView& y);
  };

  class TransformerSelfAttention : public MultiHeadAttention
  {
  public:
    TransformerSelfAttention(const Model& model, const std::string& scope, size_t num_heads);
    StorageView& operator()(const StorageView& queries,
                            const StorageView& queries_lengths,
                            StorageView* cached_keys = nullptr,
                            StorageView* cached_values = nullptr,
                            ssize_t step = 0);
  private:
    Dense _linear_in;
    Dense _linear_out;
    StorageView _queries_proj;
    StorageView _keys_proj;
    StorageView _values_proj;
    static void cache_proj(ssize_t step, StorageView& proj, StorageView& cache);
  };

  class TransformerAttention : public MultiHeadAttention
  {
  public:
    TransformerAttention(const Model& model,
                         const std::string& scope,
                         size_t num_heads);
    StorageView& operator()(const StorageView& queries,
                            const StorageView& queries_lengths,
                            const StorageView& memory,
                            const StorageView& memory_lengths,
                            StorageView* cached_keys = nullptr,
                            StorageView* cached_values = nullptr,
                            ssize_t step = -1);
  private:
    Dense _linear_query;
    Dense _linear_memory;
    Dense _linear_out;
    StorageView _keys_proj;
    StorageView _values_proj;
  };

  class TransformerEncoderLayer
  {
  public:
    TransformerEncoderLayer(const Model& model, const std::string& scope);
    StorageView& operator()(const StorageView& input,
                            const StorageView& lengths);
  private:
    TransformerSelfAttention _self_attention;
    TransformerFeedForward _ff;
  };

  class TransformerDecoderLayer
  {
  public:
    TransformerDecoderLayer(const Model& model, const std::string& scope);
    StorageView& operator()(size_t step,
                            const StorageView& input,
                            const StorageView& input_lengths,
                            const StorageView& memory,
                            const StorageView& memory_lengths,
                            StorageView& cached_self_attn_keys,
                            StorageView& cached_self_attn_values,
                            StorageView& cached_attn_keys,
                            StorageView& cached_attn_values);
  private:
    TransformerSelfAttention _self_attention;
    TransformerAttention _encoder_attention;
    TransformerFeedForward _ff;
  };

  class TransformerEncoder : public Encoder
  {
  public:
    TransformerEncoder(const Model& model, const std::string& scope);
    StorageView& encode(const StorageView& ids,
                        const StorageView& lengths) override;
  private:
    ScaledEmbeddings _scaled_embeddings;
    PositionEncoder _position_encoder;
    LayerNorm _output_norm;
    std::vector<TransformerEncoderLayer> _layers;
  };

  class TransformerDecoderState : public DecoderState {
  public:
    TransformerDecoderState(size_t num_layers);
    void reset(const StorageView& memory,
               const StorageView& memory_lengths) override;
  };

  class TransformerDecoder : public Decoder
  {
  public:
    TransformerDecoder(const Model& model, const std::string& scope);
    StorageView& logits(size_t step, const StorageView& ids) override;
  private:
    ScaledEmbeddings _scaled_embeddings;
    PositionEncoder _position_encoder;
    LayerNorm _output_norm;
    std::vector<TransformerDecoderLayer> _layers;
    Dense _proj;
  };

}
