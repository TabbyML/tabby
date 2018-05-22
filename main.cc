#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <cstring>
#include <unordered_map>
#include <chrono>

#include "model.h"
#include "vocabulary.h"
#include "routines.h"
#include "storage_view.h"
#include "ops.h"
#include "compute.h"

template <typename T>
static void pad_sequences(const onmt::StorageView<T>& flattened,
                          const onmt::StorageView<size_t>& lengths,
                          onmt::StorageView<T>& padded) {
  assert(flattened.rank() == 2);
  size_t batch_size = lengths.dim(0);
  size_t max_length = *std::max_element(lengths.data(), lengths.data() + batch_size);
  size_t depth = flattened.dim(1);
  padded.resize({batch_size, max_length, depth});
  const T* src = flattened.data();
  T* dst = padded.data();
  for (size_t i = 0; i < batch_size; ++i) {
    const size_t length = lengths[i];
    size_t count = length * depth;
    onmt::compute::copy(src, dst, count);
    dst += count;
    src += count;
    if (length < max_length) {
      count = (max_length - length) * depth;
      onmt::compute::fill(dst, static_cast<T>(0), count);
      dst += count;
    }
  }
}

template <typename T>
static void unpad_sequences(const onmt::StorageView<T>& padded,
                            const onmt::StorageView<size_t>& lengths,
                            onmt::StorageView<T>& flattened) {
  assert(padded.rank() == 3);
  size_t batch_size = lengths.dim(0);
  size_t max_length = padded.dim(1);
  size_t depth = padded.dim(2);
  size_t total_length = std::accumulate(lengths.data(), lengths.data() + batch_size, 0);
  flattened.resize({total_length, depth});
  const T* src = padded.data();
  T* dst = flattened.data();
  for (size_t i = 0; i < batch_size; ++i) {
    const size_t length = lengths[i];
    size_t count = depth * length;
    onmt::compute::copy(src, dst, count);
    dst += count;
    src += count + (max_length - length) * depth;
  }
}

template <typename U, typename V>
static void swap_middle_dims(const onmt::StorageView<U>& x, onmt::StorageView<V>& y) {
  assert(x.rank() == 4);
  size_t d0 = x.dim(0);
  size_t d1 = x.dim(1);
  size_t d2 = x.dim(2);
  size_t d3 = x.dim(3);
  y.resize({d0, d2, d1, d3});
  for (size_t i0 = 0; i0 < d0; ++i0) {
    for (size_t i1 = 0; i1 < d1; ++i1) {
      for (size_t i2 = 0; i2 < d2; ++i2) {
        for (size_t i3 = 0; i3 < d3; ++i3) {
          y[i3 + (i1 * d3) + (i2 * d3 * d1) + (i0 * d3 * d1 * d2)] =
            x[i3 + (i2 * d3) + (i1 * d3 * d2) + (i0 * d3 * d2 * d1)];
        }
      }
    }
  }
}

class ScaledEmbeddings
{
public:
  ScaledEmbeddings(const onmt::Model& model, const std::string& scope)
    : _gather_op(model.get_variable(scope + "/w_embs")) {
  }

  onmt::StorageView<float>& operator()(const onmt::StorageView<size_t>& ids) {
    _gather_op(ids, _output);
    size_t embedding_size = output_depth();
    onmt::compute::mul(static_cast<float>(sqrt(embedding_size)), _output.data(), _output.size());
    return _output;
  }

  size_t output_depth() const {
    return _gather_op.output_depth();
  }

private:
  onmt::ops::Gather<float> _gather_op;
  onmt::StorageView<float> _output;
};

class PositionEncoder
{
public:
  onmt::StorageView<float>& operator()(const onmt::StorageView<float>& input,
                                 const onmt::StorageView<size_t>& lengths) {
    assert(input.rank() == 2);
    size_t batch_size = lengths.dim(0);
    size_t depth = input.dim(1);
    if (_cached_encodings.empty())
      precompute_position_encoding(_max_cached_time, depth);
    _output.resize_as(input);
    onmt::compute::copy(input.data(), _output.data(), input.size());
    const float* x = _cached_encodings.data();
    size_t offset = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      const size_t length = lengths[i];
      float* y = _output.index({offset});
      onmt::compute::add(x, y, length * depth);
      offset += length;
    }
    return _output;
  }

  onmt::StorageView<float>& operator()(const onmt::StorageView<float>& input, size_t index) {
    onmt::StorageView<size_t> lengths({input.dim(0)}, 1);
    return operator()(input, lengths);
  }

private:
  size_t _max_cached_time = 500;
  onmt::StorageView<float> _cached_encodings;
  onmt::StorageView<float> _output;

  void precompute_position_encoding(size_t max_time, size_t depth) {
    float log_timescale_increment = log(10000) / (depth / 2 - 1);
    onmt::StorageView<float> timescales({depth / 2}, -log_timescale_increment);
    for (size_t i = 0; i < timescales.size(); ++i)
      timescales[i] = exp(timescales[i] * i);

    onmt::StorageView<float> scaled_time({max_time, depth / 2});
    for (size_t i = 0; i < scaled_time.dim(0); ++i) {
      for (size_t j = 0; j < scaled_time.dim(1); ++j) {
        scaled_time[{i, j}] = (i + 1) * timescales[j];
      }
    }

    onmt::StorageView<float> sin_encoding(scaled_time.shape());
    onmt::StorageView<float> cos_encoding(scaled_time.shape());

    vsSin(scaled_time.size(), scaled_time.data(), sin_encoding.data());
    vsCos(scaled_time.size(), scaled_time.data(), cos_encoding.data());

    _cached_encodings.resize({max_time, depth});
    concat_in_depth({sin_encoding.data(), cos_encoding.data()},
                    {depth / 2, depth / 2}, max_time,
                    _cached_encodings.data());
  }
};

class Dense
{
public:
  Dense(const onmt::Model& model, const std::string& scope)
    : _gemm(1.0, 1.0, true, false, true)
    , _weight(model.get_variable(scope + "/kernel"))
    , _bias(model.get_variable(scope + "/bias")) {
  }

  onmt::StorageView<float>& operator()(const onmt::StorageView<float>& input) {
    _gemm(input, _weight, &_bias, _output);
    return _output;
  }

  size_t output_depth() const {
    return _weight.dim(-2);
  }

private:
  onmt::ops::Gemm<float, float> _gemm;
  const onmt::StorageView<float>& _weight;
  const onmt::StorageView<float>& _bias;
  onmt::StorageView<float> _output;
};

class LayerNorm
{
public:
  LayerNorm(const onmt::Model& model, const std::string& scope)
    : _op(model.get_variable(scope + "/beta"), model.get_variable(scope + "/gamma")) {
  }

  onmt::StorageView<float>& operator()(const onmt::StorageView<float>& input) {
    _op(input, _output);
    return _output;
  }

private:
  onmt::ops::LayerNorm _op;
  onmt::StorageView<float> _output;
};

class TransformerFeedForward
{
public:
  TransformerFeedForward(const onmt::Model& model,
                         const std::string& scope)
    : _layer_norm(model, scope + "/LayerNorm")
    , _ff1(model, scope + "/conv1d")
    , _ff2(model, scope + "/conv1d_1") {
  }

  onmt::StorageView<float>& operator()(const onmt::StorageView<float>& input) {
    const onmt::StorageView<float>& normed = _layer_norm(input);
    onmt::StorageView<float>& inner = _ff1(normed);
    onmt::ops::ReLU()(inner);
    onmt::StorageView<float>& outer = _ff2(inner);
    onmt::compute::add(input.data(), outer.data(), input.size());
    return outer;
  }

private:
  LayerNorm _layer_norm;
  Dense _ff1;
  Dense _ff2;
};

class DotProductAttention
{
public:

  onmt::StorageView<float>& operator()(const onmt::StorageView<float>& queries,
                                 const onmt::StorageView<float>& keys,
                                 const onmt::StorageView<float>& values,
                                 const onmt::StorageView<size_t>& values_lengths) {
    assert(queries.rank() == 4);
    assert(keys.rank() == 4);
    assert(values.rank() == 4);

    size_t batch_size = queries.dim(0);
    size_t num_heads = queries.dim(1);
    size_t queries_time = queries.dim(2);
    size_t memory_time = keys.dim(2);

    onmt::ops::MatMul()(queries, keys, false, true, _dot);

    if (batch_size > 1) {
      for (size_t b = 0; b < batch_size; ++b) {
        const size_t length = values_lengths[b];
        if (length == memory_time)
          continue;
        for (size_t h = 0; h < num_heads; ++h) {
          for (size_t i = 0; i < queries_time; ++i) {
            float* x = _dot.index({b, h, i});
            onmt::compute::fill(x + length, std::numeric_limits<float>::lowest(), memory_time - length);
          }
        }
      }
    }

    onmt::ops::SoftMax()(_dot, _attn);
    onmt::ops::MatMul()(_attn, values, _dot);
    return _dot;
  }

private:
  onmt::StorageView<float> _dot;
  onmt::StorageView<float> _attn;
};

class MultiHeadAttention
{
public:
  MultiHeadAttention(const onmt::Model& model,
                     const std::string& scope,
                     size_t num_heads)
    : _layer_norm(model, scope + "/LayerNorm")
    , _num_heads(num_heads) {
  }

  void split_heads(const onmt::StorageView<float>& x, onmt::StorageView<float>& y) {
    assert(x.rank() == 3);
    onmt::StorageView<const float> z(x.data(), {x.dim(0), x.dim(1), _num_heads, x.dim(2) / _num_heads});
    swap_middle_dims(z, y);
  }

  void combine_heads(const onmt::StorageView<float>& x, onmt::StorageView<float>& y) {
    swap_middle_dims(x, y);
    y.reshape({y.dim(0), y.dim(1), y.dim(-1) * _num_heads});
  }

  onmt::StorageView<float>& compute_attention(const onmt::StorageView<float>& queries,
                                        const onmt::StorageView<float>& keys,
                                        const onmt::StorageView<float>& values,
                                        const onmt::StorageView<size_t>& queries_lengths,
                                        const onmt::StorageView<size_t>& values_lengths) {
    size_t dk = queries.dim(-1) / _num_heads;

    pad_sequences(queries, queries_lengths, _padded_queries);
    pad_sequences(keys, values_lengths, _padded_keys);
    pad_sequences(values, values_lengths, _padded_values);

    split_heads(_padded_queries, _split_queries);
    split_heads(_padded_keys, _split_keys);
    split_heads(_padded_values, _split_values);

    onmt::compute::mul(static_cast<float>(1.0 / sqrt(dk)), _split_queries.data(), _split_queries.size());

    const onmt::StorageView<float>& context = _attention(_split_queries,
                                                   _split_keys,
                                                   _split_values,
                                                   values_lengths);

    onmt::StorageView<float>& combined = _padded_queries;
    combine_heads(context, combined);

    onmt::StorageView<float>& combined_pruned = _padded_keys;
    unpad_sequences(combined, queries_lengths, combined_pruned);
    return combined_pruned;
  }

protected:
  LayerNorm _layer_norm;

private:
  size_t _num_heads;
  DotProductAttention _attention;
  onmt::StorageView<float> _padded_queries;
  onmt::StorageView<float> _padded_keys;
  onmt::StorageView<float> _padded_values;
  onmt::StorageView<float> _split_queries;
  onmt::StorageView<float> _split_keys;
  onmt::StorageView<float> _split_values;
};

class TransformerSelfAttention : public MultiHeadAttention
{
private:
  Dense _linear_in;
  Dense _linear_out;
  onmt::StorageView<float> _splits;

public:
  TransformerSelfAttention(const onmt::Model& model,
                           const std::string& scope,
                           size_t num_heads)
    : MultiHeadAttention(model, scope, num_heads)
    , _linear_in(model, scope + "/conv1d")
    , _linear_out(model, scope + "/conv1d_1") {
  }

  onmt::StorageView<float>& operator()(const onmt::StorageView<float>& queries,
                                 const onmt::StorageView<size_t>& queries_lengths,
                                 onmt::StorageView<float>* cached_keys = nullptr,
                                 onmt::StorageView<float>* cached_values = nullptr,
                                 ssize_t step = 0) {
    const onmt::StorageView<float>& normed_queries = _layer_norm(queries);
    const onmt::StorageView<float>& fused_proj = _linear_in(normed_queries);

    _splits.resize_as(fused_proj);
    std::vector<float*> splits = split_in_depth(fused_proj.data(),
                                                fused_proj.dim(0), fused_proj.dim(1),
                                                3, _splits.data());

    size_t split_depth = fused_proj.dim(1) / 3;
    onmt::StorageView<float> queries_proj(splits[0], {fused_proj.dim(0), split_depth});
    onmt::StorageView<float> keys_proj(splits[1], {fused_proj.dim(0), split_depth});
    onmt::StorageView<float> values_proj(splits[2], {fused_proj.dim(0), split_depth});
    onmt::StorageView<size_t> values_lengths(queries_lengths);

    if (step >= 0 && cached_keys != nullptr) {
      cache_proj(step, keys_proj, *cached_keys);
      cache_proj(step, values_proj, *cached_values);
      keys_proj.shallow_copy(*cached_keys);
      values_proj.shallow_copy(*cached_values);
      values_lengths.fill(step + 1);
    }

    const onmt::StorageView<float>& attention_output = compute_attention(queries_proj,
                                                                   keys_proj,
                                                                   values_proj,
                                                                   queries_lengths,
                                                                   values_lengths);

    onmt::StorageView<float>& output = _linear_out(attention_output);
    onmt::compute::add(queries.data(), output.data(), queries.size());
    return output;
  }

  static void cache_proj(ssize_t step, const onmt::StorageView<float>& proj, onmt::StorageView<float>& cache) {
    assert(proj.rank() == 2);
    if (step == 0) {
      cache = proj;
      return;
    }
    assert(cache.rank() == 2);
    size_t batch_size = proj.dim(0);
    size_t depth = proj.dim(1);
    static onmt::StorageView<float> tmp;
    tmp = cache;
    cache.grow(0, batch_size);
    const float* src = tmp.data();
    float* dst = cache.data();
    for (size_t i = 0; i < batch_size; ++i) {
      onmt::compute::copy(src, dst, step * depth);
      src += step * depth;
      dst += step * depth;
      onmt::compute::copy(proj.index({i}), dst, depth);
      dst += depth;
    }
  }

};

class TransformerAttention : public MultiHeadAttention
{
private:
  Dense _linear_query;
  Dense _linear_memory;
  Dense _linear_out;
  onmt::StorageView<float> _splits;

public:
  TransformerAttention(const onmt::Model& model,
                       const std::string& scope,
                       size_t num_heads)
    : MultiHeadAttention(model, scope, num_heads)
    , _linear_query(model, scope + "/conv1d")
    , _linear_memory(model, scope + "/conv1d_1")
    , _linear_out(model, scope + "/conv1d_2") {
  }

  onmt::StorageView<float>& operator()(const onmt::StorageView<float>& queries,
                                 const onmt::StorageView<size_t>& queries_lengths,
                                 const onmt::StorageView<float>& memory,
                                 const onmt::StorageView<size_t>& memory_lengths,
                                 onmt::StorageView<float>* cached_keys = nullptr,
                                 onmt::StorageView<float>* cached_values = nullptr,
                                 ssize_t step = -1) {
    size_t depth = _linear_query.output_depth();

    const onmt::StorageView<float>& normed_queries = _layer_norm(queries);
    const onmt::StorageView<float>& queries_proj = _linear_query(normed_queries);
    onmt::StorageView<float> keys_proj;
    onmt::StorageView<float> values_proj;

    if (step > 0 && cached_keys != nullptr && !cached_keys->empty()) {
      keys_proj.shallow_copy(*cached_keys);
      values_proj.shallow_copy(*cached_values);
    } else {
      const onmt::StorageView<float>& memory_proj = _linear_memory(memory);
      _splits.resize_as(memory_proj);
      std::vector<float*> splits = split_in_depth(memory_proj.data(),
                                                  memory_proj.dim(0), memory_proj.dim(1),
                                                  2, _splits.data());
      keys_proj.assign(splits[0], {memory_proj.dim(0), depth});
      values_proj.assign(splits[1], {memory_proj.dim(0), depth});
      if (cached_keys != nullptr) {
        *cached_keys = keys_proj;
        *cached_values = values_proj;
      }
    }

    const onmt::StorageView<float>& attention_output = compute_attention(queries_proj,
                                                                   keys_proj,
                                                                   values_proj,
                                                                   queries_lengths,
                                                                   memory_lengths);

    onmt::StorageView<float>& output = _linear_out(attention_output);
    onmt::compute::add(queries.data(), output.data(), queries.size());
    return output;
  }
};

class TransformerEncoderLayer
{
public:
  TransformerEncoderLayer(const onmt::Model& model,
                          const std::string& scope)
    : _self_attention(model, scope + "/multi_head", 8)
    , _ff(model, scope + "/ffn") {
  }

  onmt::StorageView<float>& operator()(const onmt::StorageView<float>& input,
                                 const onmt::StorageView<size_t>& lengths) {
    const onmt::StorageView<float>& context = _self_attention(input, lengths);
    return _ff(context);
  }

private:
  TransformerSelfAttention _self_attention;
  TransformerFeedForward _ff;
};

class TransformerDecoderLayer
{
public:
  TransformerDecoderLayer(const onmt::Model& model,
                          const std::string& scope)
    : _self_attention(model, scope + "/masked_multi_head", 8)
    , _encoder_attention(model, scope + "/multi_head", 8)
    , _ff(model, scope + "/ffn") {
  }

  onmt::StorageView<float>& operator()(size_t step,
                                 const onmt::StorageView<float>& input,
                                 const onmt::StorageView<size_t>& input_lengths,
                                 const onmt::StorageView<float>& memory,
                                 const onmt::StorageView<size_t>& memory_lengths,
                                 onmt::StorageView<float>& cached_self_attn_keys,
                                 onmt::StorageView<float>& cached_self_attn_values,
                                 onmt::StorageView<float>& cached_attn_keys,
                                 onmt::StorageView<float>& cached_attn_values) {
    const onmt::StorageView<float>& encoded = _self_attention(
      input, input_lengths, &cached_self_attn_keys, &cached_self_attn_values, step);
    const onmt::StorageView<float>& context = _encoder_attention(
      encoded, input_lengths, memory, memory_lengths, &cached_attn_keys, &cached_attn_values, step);
    return _ff(context);
  }

private:
  TransformerSelfAttention _self_attention;
  TransformerAttention _encoder_attention;
  TransformerFeedForward _ff;
};

template <typename TransformerLayer>
class TransformerStack
{
public:
  TransformerStack(const onmt::Model& model, const std::string& scope)
    : _scaled_embeddings(model, scope)
    , _position_encoder()
    , _output_norm(model, scope + "/LayerNorm") {
    for (size_t l = 0;; ++l) {
      try {
        _layers.emplace_back(model, scope + "/layer_" + std::to_string(l));
      } catch (std::exception&) {
        break;
      }
    }
  }

protected:
  ScaledEmbeddings _scaled_embeddings;
  PositionEncoder _position_encoder;
  LayerNorm _output_norm;
  std::vector<TransformerLayer> _layers;
};

class TransformerEncoder : public TransformerStack<TransformerEncoderLayer>
{
public:
  TransformerEncoder(const onmt::Model& model, const std::string& scope)
    : TransformerStack(model, scope) {
  }

  onmt::StorageView<float>& operator()(const onmt::StorageView<size_t>& ids,
                                 const onmt::StorageView<size_t>& lengths) {
    const onmt::StorageView<float>& embeddings = _scaled_embeddings(ids);
    const onmt::StorageView<float>& input = _position_encoder(embeddings, lengths);

    const onmt::StorageView<float>* x = &input;
    for (auto& layer : _layers) {
      x = &layer(*x, lengths);
    }
    return _output_norm(*x);
  }
};

struct TransformerDecoderState {
  onmt::StorageView<float> memory;
  onmt::StorageView<size_t> memory_lengths;
  std::vector<onmt::StorageView<float>> cache;
};

static void remove_batch(onmt::StorageView<float>& s,
                         const onmt::StorageView<size_t>& lengths,
                         const std::vector<bool>& finished) {
  assert(s.rank() == 2);
  static onmt::StorageView<float> tmp;
  tmp = s;
  size_t batch_size = lengths.dim(0);
  size_t depth = s.dim(1);
  const float* src = tmp.data();
  float* dst = s.data();
  size_t cum_length = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    const size_t length = lengths[i];
    const size_t count = length * depth;
    if (!finished[i]) {
      onmt::compute::copy(src, dst, count);
      dst += count;
      cum_length += length;
    }
    src += count;
  }
  s.resize(0, cum_length);
}

static void remove_batch(onmt::StorageView<size_t>& s, const std::vector<bool>& finished) {
  assert(s.rank() == 1);
  size_t write_index = 0;
  size_t read_index = 0;
  while (read_index < s.dim(0)) {
    if (!finished[read_index])
      s[write_index++] = s[read_index];
    read_index++;
  }
  s.resize(0, write_index);
}

class TransformerDecoder : public TransformerStack<TransformerDecoderLayer>
{
public:
  TransformerDecoder(const onmt::Model& model, const std::string& scope)
    : TransformerStack(model, scope)
    , _proj(model, scope + "/dense") {
    _state.cache.resize(_layers.size() * 4);
  }

  void reset_state(const onmt::StorageView<float>& memory,
                   const onmt::StorageView<size_t>& memory_lengths) {
    _state.memory = memory;
    _state.memory_lengths = memory_lengths;
  }

  void prune_batch(size_t step, const std::vector<bool>& ids) {
    size_t batch_size = _state.memory_lengths.dim(0);
    onmt::StorageView<size_t> step_lengths({batch_size}, step + 1);
    for (size_t l = 0; l < _layers.size(); ++l) {
      remove_batch(_state.cache[0 + l * 4], step_lengths, ids);
      remove_batch(_state.cache[1 + l * 4], step_lengths, ids);
      remove_batch(_state.cache[2 + l * 4], _state.memory_lengths, ids);
      remove_batch(_state.cache[3 + l * 4], _state.memory_lengths, ids);
    }
    remove_batch(_state.memory, _state.memory_lengths, ids);
    remove_batch(_state.memory_lengths, ids);
  }

  onmt::StorageView<float>& operator()(size_t step, const onmt::StorageView<size_t>& ids) {
    size_t batch_size = ids.dim(0);
    const onmt::StorageView<float>& embeddings = _scaled_embeddings(ids);
    const onmt::StorageView<float>& input = _position_encoder(embeddings, step);
    onmt::StorageView<size_t> query_lengths({batch_size}, 1);

    const onmt::StorageView<float>* x = &input;
    for (size_t l = 0; l < _layers.size(); ++l) {
      x = &_layers[l](step,
                      *x,
                      query_lengths,
                      _state.memory,
                      _state.memory_lengths,
                      _state.cache[0 + l * 4],
                      _state.cache[1 + l * 4],
                      _state.cache[2 + l * 4],
                      _state.cache[3 + l * 4]);
    }
    const onmt::StorageView<float>& normed = _output_norm(*x);
    return _proj(normed);
  }

private:
  Dense _proj;
  TransformerDecoderState _state;
};

void translate(const std::vector<std::vector<std::string> >& input_tokens,
               const onmt::Vocabulary& vocabulary,
               TransformerEncoder& encoder,
               TransformerDecoder& decoder) {
  onmt::StorageView<size_t> lengths({input_tokens.size()});
  size_t total_length = 0;
  for (size_t i = 0; i < input_tokens.size(); ++i) {
    const size_t length = input_tokens[i].size();
    lengths[i] = length;
    total_length += length;
  }

  onmt::StorageView<size_t> ids({total_length});
  size_t offset = 0;
  for (const auto& tokens : input_tokens) {
    for (const auto& token : tokens) {
      ids[offset] = vocabulary.to_id(token);
      offset += 1;
    }
  }

  // for (auto id : ids)
  //   std::cout << id << std::endl;

  size_t batch_size = lengths.size();
  const onmt::StorageView<float>& encoded = encoder(ids, lengths);

  decoder.reset_state(encoded, lengths);

  onmt::StorageView<size_t> sample_from({batch_size}, vocabulary.to_id("<s>"));
  onmt::StorageView<float> probs({batch_size, vocabulary.size()});
  std::vector<std::vector<size_t> > sampled_ids(batch_size);
  std::vector<bool> finished(batch_size, false);
  std::vector<size_t> batch_offset(batch_size);
  for (size_t i = 0; i < batch_offset.size(); ++i)
    batch_offset[i] = i;
  size_t max_steps = 200;

  for (size_t step = 0; step < max_steps; ++step) {
    onmt::StorageView<float>& logits = decoder(step, sample_from);
    onmt::ops::SoftMax()(logits, probs);

    std::vector<bool> finished_batch(logits.dim(0), false);
    bool one_finished = false;
    for (size_t i = 0; i < logits.dim(0); ++i) {
      size_t best = onmt::compute::max_element(probs.index({i}), vocabulary.size());
      size_t batch_id = batch_offset[i];
      if (best == 2) {
        finished[batch_id] = true;
        finished_batch[i] = true;
        one_finished = true;
      } else {
        sample_from[i] = best;
        sampled_ids[batch_id].push_back(best);
      }
    }

    if (one_finished) {
      remove_batch(sample_from, finished_batch);
      if (sample_from.empty())
        break;
      decoder.prune_batch(step, finished_batch);
      size_t write_index = 0;
      size_t read_index = 0;
      for (; read_index < finished_batch.size(); ++read_index) {
        if (!finished_batch[read_index])
          batch_offset[write_index++] = batch_offset[read_index];
      }
    }
  }

  for (size_t i = 0; i < batch_size; ++i) {
    for (auto id : sampled_ids[i]) {
      std::cout << " " << vocabulary.to_token(id);
    }
    std::cout << std::endl;
  }
}

int main(int argc, char* argv[]) {
  vmlSetMode(VML_EP);

  onmt::Model model("/home/klein/dev/ctransformer/model.bin");
  onmt::Vocabulary vocabulary("/home/klein/data/wmt-ende/wmtende.vocab");

  TransformerEncoder encoder(model, "transformer/encoder");
  TransformerDecoder decoder(model, "transformer/decoder");

  std::ifstream text_file("/home/klein/data/wmt-ende/valid.en.200");
  std::vector<std::vector<std::string> > input_tokens;
  std::string line;

  size_t max_batch_size = argc > 1 ? std::stoi(argv[1]) : 1;
  size_t num_tokens = 0;

  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

  while (std::getline(text_file, line)) {
    input_tokens.emplace_back();
    std::string token;
    for (size_t i = 0; i < line.length(); ++i) {
      if (line[i] == ' ') {
        if (!token.empty()) {
          input_tokens.back().push_back(token);
          token.clear();
        }
      } else {
        token += line[i];
      }
    }
    if (!token.empty()) {
      input_tokens.back().push_back(token);
      token.clear();
    }
    num_tokens += input_tokens.back().size();

    if (input_tokens.size() == max_batch_size) {
      translate(input_tokens, vocabulary, encoder, decoder);
      input_tokens.clear();
    }
  }

  if (!input_tokens.empty()) {
    translate(input_tokens, vocabulary, encoder, decoder);
  }

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  std::cerr << static_cast<double>(num_tokens) / static_cast<double>(duration / 1000) << std::endl;
  return 0;
}
