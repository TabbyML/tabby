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

template <typename T>
static void pad_sequences(const StorageView<T>& flattened,
                          const StorageView<size_t>& lengths,
                          StorageView<T>& padded) {
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
    array_copy(src, dst, count);
    dst += count;
    src += count;
    if (length < max_length) {
      count = (max_length - length) * depth;
      array_fill(dst, 0, count);
      dst += count;
    }
  }
}

template <typename T>
static void unpad_sequences(const StorageView<T>& padded,
                            const StorageView<size_t>& lengths,
                            StorageView<T>& flattened) {
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
    array_copy(src, dst, count);
    dst += count;
    src += count + (max_length - length) * depth;
  }
}

template <typename U, typename V>
static void swap_middle_dims(const StorageView<U>& x, StorageView<V>& y) {
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
  ScaledEmbeddings(const Model& model, const std::string& scope)
    : _weight(model.get_variable(scope + "/w_embs")) {
  }

  StorageView<float>& operator()(const StorageView<size_t>& ids) {
    size_t batch_size = ids.dim(0);
    size_t embedding_size = output_depth();
    _output.resize({batch_size, embedding_size});
    gather(ids.data(), _weight.data(), batch_size, embedding_size, _output.data());
    array_mul(sqrt(embedding_size), _output.data(), _output.size());
    return _output;
  }

  size_t output_depth() const {
    return _weight.shape().back();
  }

private:
  const Variable& _weight;
  StorageView<float> _output;
};

class PositionEncoder
{
public:
  StorageView<float>& operator()(const StorageView<float>& input, size_t index) {
    assert(input.rank() == 2);
    size_t batch_size = input.dim(0);
    size_t depth = input.dim(1);
    if (_cached_encodings.empty())
      precompute_position_encoding(_max_cached_time, depth);
    _output.resize_as(input);
    array_copy(input.data(), _output.data(), input.size());
    const float* x = _cached_encodings.index({index});
    for (size_t i = 0; i < batch_size; ++i) {
      float* y = _output.index({i});
      array_add(x, y, depth);
    }
    return _output;
  }

  StorageView<float>& operator()(const StorageView<float>& input,
                                 const StorageView<size_t>& lengths) {
    assert(input.rank() == 2);
    size_t batch_size = lengths.dim(0);
    size_t depth = input.dim(1);
    if (_cached_encodings.empty())
      precompute_position_encoding(_max_cached_time, depth);
    _output.resize_as(input);
    array_copy(input.data(), _output.data(), input.size());
    const float* x = _cached_encodings.data();
    size_t offset = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      const size_t length = lengths[i];
      float* y = _output.index({offset});
      array_add(x, y, length * depth);
      offset += length;
    }
    return _output;
  }

private:
  size_t _max_cached_time = 500;
  StorageView<float> _cached_encodings;
  StorageView<float> _output;

  void precompute_position_encoding(size_t max_time, size_t depth) {
    float log_timescale_increment = log(10000) / (depth / 2 - 1);
    StorageView<float> timescales({depth / 2}, -log_timescale_increment);
    for (size_t i = 0; i < timescales.size(); ++i)
      timescales[i] = exp(timescales[i] * i);

    StorageView<float> scaled_time({max_time, depth / 2});
    for (size_t i = 0; i < scaled_time.dim(0); ++i) {
      for (size_t j = 0; j < scaled_time.dim(1); ++j) {
        scaled_time[{i, j}] = (i + 1) * timescales[j];
      }
    }

    StorageView<float> sin_encoding(scaled_time.shape());
    StorageView<float> cos_encoding(scaled_time.shape());

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
  Dense(const Model& model, const std::string& scope)
    : _weight(model.get_variable(scope + "/kernel"))
    , _bias(model.get_variable(scope + "/bias")) {
  }

  StorageView<float>& operator()(const StorageView<float>& input) {
    const auto& shape = input.shape();
    size_t batch_size = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i)
      batch_size *= shape[i];
    size_t in_depth = shape.back();
    size_t out_depth = output_depth();
    _output.resize({batch_size, out_depth});
    linear(input.data(), _weight.data(), _bias.data(),
           batch_size, in_depth, out_depth, _output.data());
    return _output;
  }

  size_t output_depth() const {
    return _weight.shape().back();
  }

private:
  const Variable& _weight;
  const Variable& _bias;
  StorageView<float> _output;
};

class LayerNorm
{
public:
  LayerNorm(const Model& model, const std::string& scope)
    : _beta(model.get_variable(scope + "/beta"))
    , _gamma(model.get_variable(scope + "/gamma")) {
  }

  StorageView<float>& operator()(const StorageView<float>& input) {
    assert(input.rank() == 2);
    size_t batch_size = input.dim(0);
    size_t depth = input.dim(1);
    _output.resize_as(input);
    _tmp.resize({depth});
    for (size_t i = 0; i < batch_size; ++i) {
      const float* x = input.index({i});
      float* y = _output.index({i});
      float mean = array_mean(x, depth);
      array_copy(x, y, depth);
      array_sub(mean, y, depth); // y is now centered
      array_pow(y, _tmp.data(), 2, depth);
      float variance = array_mean(_tmp.data(), depth);
      array_mul(1.0 / sqrt(variance + EPSILON), y, depth); // y is now centered and normalized.
      array_mul(_gamma.data(), y, depth);
      array_add(_beta.data(), y, depth);
    }
    return _output;
  }

private:
  const Variable& _beta;
  const Variable& _gamma;
  StorageView<float> _tmp;
  StorageView<float> _output;
};

class TransformerFeedForward
{
public:
  TransformerFeedForward(const Model& model,
                         const std::string& scope)
    : _layer_norm(model, scope + "/LayerNorm")
    , _ff1(model, scope + "/conv1d")
    , _ff2(model, scope + "/conv1d_1") {
  }

  StorageView<float>& operator()(const StorageView<float>& input) {
    const StorageView<float>& normed = _layer_norm(input);
    StorageView<float>& inner = _ff1(normed);
    relu(inner.data(), inner.size());
    StorageView<float>& outer = _ff2(inner);
    array_add(input.data(), outer.data(), input.size());
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

  StorageView<float>& operator()(const StorageView<float>& queries,
                                 const StorageView<float>& keys,
                                 const StorageView<float>& values,
                                 const StorageView<size_t>& values_lengths) {
    assert(queries.rank() == 4);
    assert(keys.rank() == 4);
    assert(values.rank() == 4);

    size_t batch_size = queries.dim(0);
    size_t num_heads = queries.dim(1);
    size_t queries_time = queries.dim(2);
    size_t memory_time = keys.dim(2);
    size_t depth = queries.dim(3);

    _dot.resize({batch_size, num_heads, queries_time, memory_time});
    _attn.resize_as(_dot);

    batch_mat_mul(queries.data(), keys.data(),
                  CblasNoTrans, CblasTrans,
                  batch_size * num_heads, queries_time, memory_time, depth,
                  _dot.data());

    if (batch_size > 1) {
      for (size_t b = 0; b < batch_size; ++b) {
        const size_t length = values_lengths[b];
        if (length == memory_time)
          continue;
        for (size_t h = 0; h < num_heads; ++h) {
          for (size_t i = 0; i < queries_time; ++i) {
            float* x = _dot.index({b, h, i});
            array_fill(x + length, std::numeric_limits<float>::lowest(), memory_time - length);
          }
        }
      }
    }

    softmax(_dot.data(), batch_size * num_heads * queries_time, memory_time, _attn.data());

    StorageView<float>& output = _dot;
    output.resize_as(queries);
    batch_mat_mul(_attn.data(), values.data(),
                  CblasNoTrans, CblasNoTrans,
                  batch_size * num_heads, queries_time, depth, memory_time,
                  output.data());
    return output;
  }

private:
  StorageView<float> _dot;
  StorageView<float> _attn;
};

class MultiHeadAttention
{
public:
  MultiHeadAttention(const Model& model,
                     const std::string& scope,
                     size_t num_heads)
    : _layer_norm(model, scope + "/LayerNorm")
    , _num_heads(num_heads) {
  }

  void split_heads(const StorageView<float>& x, StorageView<float>& y) {
    assert(x.rank() == 3);
    StorageView<const float> z(x.data(), {x.dim(0), x.dim(1), _num_heads, x.dim(2) / _num_heads});
    swap_middle_dims(z, y);
  }

  void combine_heads(const StorageView<float>& x, StorageView<float>& y) {
    swap_middle_dims(x, y);
    y.reshape({y.dim(0), y.dim(1), y.dim(-1) * _num_heads});
  }

  StorageView<float>& compute_attention(const StorageView<float>& queries,
                                        const StorageView<float>& keys,
                                        const StorageView<float>& values,
                                        const StorageView<size_t>& queries_lengths,
                                        const StorageView<size_t>& values_lengths) {
    size_t dk = queries.dim(-1) / _num_heads;

    pad_sequences(queries, queries_lengths, _padded_queries);
    pad_sequences(keys, values_lengths, _padded_keys);
    pad_sequences(values, values_lengths, _padded_values);

    split_heads(_padded_queries, _split_queries);
    split_heads(_padded_keys, _split_keys);
    split_heads(_padded_values, _split_values);

    array_mul(1.0 / sqrt(dk), _split_queries.data(), _split_queries.size());

    const StorageView<float>& context = _attention(_split_queries,
                                                   _split_keys,
                                                   _split_values,
                                                   values_lengths);

    StorageView<float>& combined = _padded_queries;
    combine_heads(context, combined);

    StorageView<float>& combined_pruned = _padded_keys;
    unpad_sequences(combined, queries_lengths, combined_pruned);
    return combined_pruned;
  }

protected:
  LayerNorm _layer_norm;

private:
  size_t _num_heads;
  DotProductAttention _attention;
  StorageView<float> _padded_queries;
  StorageView<float> _padded_keys;
  StorageView<float> _padded_values;
  StorageView<float> _split_queries;
  StorageView<float> _split_keys;
  StorageView<float> _split_values;
};

class TransformerSelfAttention : public MultiHeadAttention
{
private:
  Dense _linear_in;
  Dense _linear_out;
  StorageView<float> _splits;

public:
  TransformerSelfAttention(const Model& model,
                           const std::string& scope,
                           size_t num_heads)
    : MultiHeadAttention(model, scope, num_heads)
    , _linear_in(model, scope + "/conv1d")
    , _linear_out(model, scope + "/conv1d_1") {
  }

  StorageView<float>& operator()(const StorageView<float>& queries,
                                 const StorageView<size_t>& queries_lengths,
                                 StorageView<float>* cached_keys = nullptr,
                                 StorageView<float>* cached_values = nullptr,
                                 ssize_t step = 0) {
    const StorageView<float>& normed_queries = _layer_norm(queries);
    const StorageView<float>& fused_proj = _linear_in(normed_queries);

    _splits.resize_as(fused_proj);
    std::vector<float*> splits = split_in_depth(fused_proj.data(),
                                                fused_proj.dim(0), fused_proj.dim(1),
                                                3, _splits.data());

    size_t split_depth = fused_proj.dim(1) / 3;
    StorageView<float> queries_proj(splits[0], {fused_proj.dim(0), split_depth});
    StorageView<float> keys_proj(splits[1], {fused_proj.dim(0), split_depth});
    StorageView<float> values_proj(splits[2], {fused_proj.dim(0), split_depth});
    StorageView<size_t> values_lengths(queries_lengths);

    if (step >= 0 && cached_keys != nullptr) {
      cache_proj(step, keys_proj, *cached_keys);
      cache_proj(step, values_proj, *cached_values);
      keys_proj.shallow_copy(*cached_keys);
      values_proj.shallow_copy(*cached_values);
      values_lengths.fill(step + 1);
    }

    const StorageView<float>& attention_output = compute_attention(queries_proj,
                                                                   keys_proj,
                                                                   values_proj,
                                                                   queries_lengths,
                                                                   values_lengths);

    StorageView<float>& output = _linear_out(attention_output);
    array_add(queries.data(), output.data(), queries.size());
    return output;
  }

  static void cache_proj(ssize_t step, const StorageView<float>& proj, StorageView<float>& cache) {
    assert(proj.rank() == 2);
    if (step == 0) {
      cache = proj;
      return;
    }
    assert(cache.rank() == 2);
    size_t batch_size = proj.dim(0);
    size_t depth = proj.dim(1);
    static StorageView<float> tmp;
    tmp = cache;
    cache.grow(0, batch_size);
    const float* src = tmp.data();
    float* dst = cache.data();
    for (size_t i = 0; i < batch_size; ++i) {
      array_copy(src, dst, step * depth);
      src += step * depth;
      dst += step * depth;
      array_copy(proj.index({i}), dst, depth);
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
  StorageView<float> _splits;

public:
  TransformerAttention(const Model& model,
                       const std::string& scope,
                       size_t num_heads)
    : MultiHeadAttention(model, scope, num_heads)
    , _linear_query(model, scope + "/conv1d")
    , _linear_memory(model, scope + "/conv1d_1")
    , _linear_out(model, scope + "/conv1d_2") {
  }

  StorageView<float>& operator()(const StorageView<float>& queries,
                                 const StorageView<size_t>& queries_lengths,
                                 const StorageView<float>& memory,
                                 const StorageView<size_t>& memory_lengths,
                                 StorageView<float>* cached_keys = nullptr,
                                 StorageView<float>* cached_values = nullptr,
                                 ssize_t step = -1) {
    size_t depth = _linear_query.output_depth();

    const StorageView<float>& normed_queries = _layer_norm(queries);
    const StorageView<float>& queries_proj = _linear_query(normed_queries);
    StorageView<float> keys_proj;
    StorageView<float> values_proj;

    if (step > 0 && cached_keys != nullptr && !cached_keys->empty()) {
      keys_proj.shallow_copy(*cached_keys);
      values_proj.shallow_copy(*cached_values);
    } else {
      const StorageView<float>& memory_proj = _linear_memory(memory);
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

    const StorageView<float>& attention_output = compute_attention(queries_proj,
                                                                   keys_proj,
                                                                   values_proj,
                                                                   queries_lengths,
                                                                   memory_lengths);

    StorageView<float>& output = _linear_out(attention_output);
    array_add(queries.data(), output.data(), queries.size());
    return output;
  }
};

class TransformerEncoderLayer
{
public:
  TransformerEncoderLayer(const Model& model,
                          const std::string& scope)
    : _self_attention(model, scope + "/multi_head", 8)
    , _ff(model, scope + "/ffn") {
  }

  StorageView<float>& operator()(const StorageView<float>& input,
                                 const StorageView<size_t>& lengths) {
    const StorageView<float>& context = _self_attention(input, lengths);
    return _ff(context);
  }

private:
  TransformerSelfAttention _self_attention;
  TransformerFeedForward _ff;
};

class TransformerDecoderLayer
{
public:
  TransformerDecoderLayer(const Model& model,
                          const std::string& scope)
    : _self_attention(model, scope + "/masked_multi_head", 8)
    , _encoder_attention(model, scope + "/multi_head", 8)
    , _ff(model, scope + "/ffn") {
  }

  StorageView<float>& operator()(size_t step,
                                 const StorageView<float>& input,
                                 const StorageView<size_t>& input_lengths,
                                 const StorageView<float>& memory,
                                 const StorageView<size_t>& memory_lengths,
                                 StorageView<float>& cached_self_attn_keys,
                                 StorageView<float>& cached_self_attn_values,
                                 StorageView<float>& cached_attn_keys,
                                 StorageView<float>& cached_attn_values) {
    const StorageView<float>& encoded = _self_attention(
      input, input_lengths, &cached_self_attn_keys, &cached_self_attn_values, step);
    const StorageView<float>& context = _encoder_attention(
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
  TransformerStack(const Model& model, const std::string& scope)
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
  TransformerEncoder(const Model& model, const std::string& scope)
    : TransformerStack(model, scope) {
  }

  StorageView<float>& operator()(const StorageView<size_t>& ids,
                                 const StorageView<size_t>& lengths) {
    const StorageView<float>& embeddings = _scaled_embeddings(ids);
    const StorageView<float>& input = _position_encoder(embeddings, lengths);

    const StorageView<float>* x = &input;
    for (auto& layer : _layers) {
      x = &layer(*x, lengths);
    }
    return _output_norm(*x);
  }
};

struct TransformerDecoderState {
  StorageView<float> memory;
  StorageView<size_t> memory_lengths;
  std::vector<StorageView<float>> cache;
};

static void remove_batch(StorageView<float>& s,
                         const StorageView<size_t>& lengths,
                         const std::vector<bool>& finished) {
  assert(s.rank() == 2);
  static StorageView<float> tmp;
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
      array_copy(src, dst, count);
      dst += count;
      cum_length += length;
    }
    src += count;
  }
  s.resize(0, cum_length);
}

static void remove_batch(StorageView<size_t>& s, const std::vector<bool>& finished) {
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
  TransformerDecoder(const Model& model, const std::string& scope)
    : TransformerStack(model, scope)
    , _proj(model, scope + "/dense") {
    _state.cache.resize(_layers.size() * 4);
  }

  void reset_state(const StorageView<float>& memory,
                   const StorageView<size_t>& memory_lengths) {
    _state.memory = memory;
    _state.memory_lengths = memory_lengths;
  }

  void prune_batch(size_t step, const std::vector<bool>& ids) {
    size_t batch_size = _state.memory_lengths.dim(0);
    StorageView<size_t> step_lengths({batch_size}, step + 1);
    for (size_t l = 0; l < _layers.size(); ++l) {
      remove_batch(_state.cache[0 + l * 4], step_lengths, ids);
      remove_batch(_state.cache[1 + l * 4], step_lengths, ids);
      remove_batch(_state.cache[2 + l * 4], _state.memory_lengths, ids);
      remove_batch(_state.cache[3 + l * 4], _state.memory_lengths, ids);
    }
    remove_batch(_state.memory, _state.memory_lengths, ids);
    remove_batch(_state.memory_lengths, ids);
  }

  StorageView<float>& operator()(size_t step, const StorageView<size_t>& ids) {
    size_t batch_size = ids.dim(0);
    const StorageView<float>& embeddings = _scaled_embeddings(ids);
    const StorageView<float>& input = _position_encoder(embeddings, step);
    StorageView<size_t> query_lengths({batch_size}, 1);

    const StorageView<float>* x = &input;
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
    const StorageView<float>& normed = _output_norm(*x);
    return _proj(normed);
  }

private:
  Dense _proj;
  TransformerDecoderState _state;
};

void translate(const std::vector<std::vector<std::string> >& input_tokens,
               const Vocabulary& vocabulary,
               TransformerEncoder& encoder,
               TransformerDecoder& decoder) {
  StorageView<size_t> lengths({input_tokens.size()});
  size_t total_length = 0;
  for (size_t i = 0; i < input_tokens.size(); ++i) {
    const size_t length = input_tokens[i].size();
    lengths[i] = length;
    total_length += length;
  }

  StorageView<size_t> ids({total_length});
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
  const StorageView<float>& encoded = encoder(ids, lengths);

  decoder.reset_state(encoded, lengths);

  StorageView<size_t> sample_from({batch_size}, vocabulary.to_id("<s>"));
  StorageView<float> probs({batch_size, vocabulary.size()});
  std::vector<std::vector<size_t> > sampled_ids(batch_size);
  std::vector<bool> finished(batch_size, false);
  std::vector<size_t> batch_offset(batch_size);
  for (size_t i = 0; i < batch_offset.size(); ++i)
    batch_offset[i] = i;
  size_t max_steps = 200;

  for (size_t step = 0; step < max_steps; ++step) {
    StorageView<float>& logits = decoder(step, sample_from);
    softmax(logits.data(), logits.dim(0), logits.dim(1), probs.data());

    std::vector<bool> finished_batch(logits.dim(0), false);
    bool one_finished = false;
    for (size_t i = 0; i < logits.dim(0); ++i) {
      size_t best = array_max_element(probs.index({i}), vocabulary.size());
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

  Model model("/home/klein/dev/ctransformer/model.bin");
  Vocabulary vocabulary("/home/klein/data/wmt-ende/wmtende.vocab");

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
