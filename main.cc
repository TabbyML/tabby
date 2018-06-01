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

class ScaledEmbeddings
{
public:
  ScaledEmbeddings(const onmt::Model& model, const std::string& scope)
    : _embeddings(model.get_variable(scope + "/w_embs"))
    , _gathered(_embeddings.dtype()) {
  }

  onmt::StorageView& operator()(const onmt::StorageView& ids) {
    size_t embedding_size = output_depth();
    if (_embeddings.dtype() == onmt::DataType::DT_FLOAT) {
      _gather_op(_embeddings, ids, _output);
    } else {
      _gather_op(_embeddings, ids, _gathered);
      onmt::ops::Unquantize(1000)(_gathered, _output);
    }
    onmt::compute::mul(static_cast<float>(sqrt(embedding_size)),
                       _output.data<float>(),
                       _output.size());
    return _output;
  }

  size_t output_depth() const {
    return _embeddings.dim(-1);
  }

private:
  onmt::ops::Gather _gather_op;
  const onmt::StorageView& _embeddings;
  onmt::StorageView _gathered;
  onmt::StorageView _output;
};

class PositionEncoder
{
public:
  onmt::StorageView& operator()(const onmt::StorageView& input,
                                const onmt::StorageView& lengths) {
    assert(input.rank() == 2);
    size_t batch_size = lengths.dim(0);
    size_t depth = input.dim(1);
    if (_cached_encodings.empty())
      precompute_position_encoding(_max_cached_time, depth);
    _output.resize_as(input);
    onmt::compute::copy(input.data<float>(), _output.data<float>(), input.size());
    const auto* x = _cached_encodings.data<float>();
    size_t offset = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      const auto length = lengths.data<int32_t>()[i];
      auto* y = _output.index<float>({offset});
      onmt::compute::add(x, y, length * depth);
      offset += length;
    }
    return _output;
  }

  onmt::StorageView& operator()(const onmt::StorageView& input, size_t index) {
    assert(input.rank() == 2);
    size_t batch_size = input.dim(0);
    size_t depth = input.dim(1);
    if (_cached_encodings.empty())
      precompute_position_encoding(_max_cached_time, depth);
    _output.resize_as(input);
    onmt::compute::copy(input.data<float>(), _output.data<float>(), input.size());
    const auto* x = _cached_encodings.index<float>({index});
    for (size_t i = 0; i < batch_size; ++i) {
      auto* y = _output.index<float>({i});
      onmt::compute::add(x, y, depth);
    }
    return _output;
  }

private:
  size_t _max_cached_time = 500;
  onmt::StorageView _cached_encodings;
  onmt::StorageView _output;

  void precompute_position_encoding(size_t max_time, size_t depth) {
    float log_timescale_increment = log(10000) / (depth / 2 - 1);
    onmt::StorageView timescales({depth / 2}, -log_timescale_increment);
    for (size_t i = 0; i < timescales.size(); ++i)
      timescales.data<float>()[i] = exp(timescales.data<float>()[i] * i);

    onmt::StorageView scaled_time({max_time, depth / 2});
    for (size_t i = 0; i < scaled_time.dim(0); ++i) {
      for (size_t j = 0; j < scaled_time.dim(1); ++j) {
        *scaled_time.index<float>({i, j}) = (i + 1) * timescales.data<float>()[j];
      }
    }

    onmt::StorageView sin_encoding;
    onmt::StorageView cos_encoding;

    onmt::ops::Sin()(scaled_time, sin_encoding);
    onmt::ops::Cos()(scaled_time, cos_encoding);
    onmt::ops::Concat(-1)({&sin_encoding, &cos_encoding}, _cached_encodings);
  }
};

class Dense
{
public:
  Dense(const onmt::Model& model, const std::string& scope)
    : _gemm_op(1, 1, true, false, true)
    , _weight(model.get_variable(scope + "/kernel"))
    , _bias(model.get_variable(scope + "/bias")) {
  }

  onmt::StorageView& operator()(const onmt::StorageView& input) {
    _gemm_op(input, _weight, _bias, _output);
    return _output;
  }

  size_t output_depth() const {
    return _weight.dim(-2);
  }

private:
  onmt::ops::Gemm _gemm_op;
  const onmt::StorageView& _weight;
  const onmt::StorageView& _bias;
  onmt::StorageView _output;
};

class LayerNorm
{
public:
  LayerNorm(const onmt::Model& model, const std::string& scope)
    : _beta(model.get_variable(scope + "/beta"))
    , _gamma(model.get_variable(scope + "/gamma")) {
  }

  onmt::StorageView& operator()(const onmt::StorageView& input) {
    _norm_op(_beta, _gamma, input, _output);
    return _output;
  }

private:
  onmt::ops::LayerNorm _norm_op;
  const onmt::StorageView& _beta;
  const onmt::StorageView& _gamma;
  onmt::StorageView _output;
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

  onmt::StorageView& operator()(const onmt::StorageView& input) {
    const onmt::StorageView& normed = _layer_norm(input);
    onmt::StorageView& inner = _ff1(normed);
    onmt::ops::ReLU()(inner);
    onmt::StorageView& outer = _ff2(inner);
    onmt::compute::add(input.data<float>(), outer.data<float>(), input.size());
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

  onmt::StorageView& operator()(const onmt::StorageView& queries,
                                const onmt::StorageView& keys,
                                const onmt::StorageView& values,
                                const onmt::StorageView& values_lengths) {
    assert(queries.rank() == 4);
    assert(keys.rank() == 4);
    assert(values.rank() == 4);

    size_t batch_size = queries.dim(0);
    size_t num_heads = queries.dim(1);
    size_t queries_time = queries.dim(2);
    size_t memory_time = keys.dim(2);

    onmt::ops::MatMul(false, true)(queries, keys, _dot);

    if (batch_size > 1) {
      for (size_t b = 0; b < batch_size; ++b) {
        const size_t length = values_lengths.data<int32_t>()[b];
        if (length == memory_time)
          continue;
        for (size_t h = 0; h < num_heads; ++h) {
          for (size_t i = 0; i < queries_time; ++i) {
            auto* x = _dot.index<float>({b, h, i});
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
  onmt::StorageView _dot;
  onmt::StorageView _attn;
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

  void split_heads(const onmt::StorageView& x, onmt::StorageView& y) {
    assert(x.rank() == 3);
    onmt::StorageView z(const_cast<float*>(x.data<float>()), {x.dim(0), x.dim(1), _num_heads, x.dim(2) / _num_heads});
    swap_middle_dims(z, y);
  }

  void combine_heads(const onmt::StorageView& x, onmt::StorageView& y) {
    swap_middle_dims(x, y);
    y.reshape({y.dim(0), y.dim(1), y.dim(-1) * _num_heads});
  }

  onmt::StorageView& compute_attention(const onmt::StorageView& queries,
                                       const onmt::StorageView& keys,
                                       const onmt::StorageView& values,
                                       const onmt::StorageView& queries_lengths,
                                       const onmt::StorageView& values_lengths) {
    size_t dk = queries.dim(-1) / _num_heads;

    pad_sequences(queries, queries_lengths, _padded_queries);
    pad_sequences(keys, values_lengths, _padded_keys);
    pad_sequences(values, values_lengths, _padded_values);

    split_heads(_padded_queries, _split_queries);
    split_heads(_padded_keys, _split_keys);
    split_heads(_padded_values, _split_values);

    onmt::compute::mul(static_cast<float>(1.0 / sqrt(dk)), _split_queries.data<float>(), _split_queries.size());

    const onmt::StorageView& context = _attention(_split_queries,
                                                  _split_keys,
                                                  _split_values,
                                                  values_lengths);

    onmt::StorageView& combined = _padded_queries;
    combine_heads(context, combined);

    onmt::StorageView& combined_pruned = _padded_keys;
    unpad_sequences(combined, queries_lengths, combined_pruned);
    return combined_pruned;
  }

protected:
  LayerNorm _layer_norm;

private:
  size_t _num_heads;
  DotProductAttention _attention;
  onmt::StorageView _padded_queries;
  onmt::StorageView _padded_keys;
  onmt::StorageView _padded_values;
  onmt::StorageView _split_queries;
  onmt::StorageView _split_keys;
  onmt::StorageView _split_values;
};

class TransformerSelfAttention : public MultiHeadAttention
{
private:
  Dense _linear_in;
  Dense _linear_out;
  onmt::StorageView _queries_proj;
  onmt::StorageView _keys_proj;
  onmt::StorageView _values_proj;

public:
  TransformerSelfAttention(const onmt::Model& model,
                           const std::string& scope,
                           size_t num_heads)
    : MultiHeadAttention(model, scope, num_heads)
    , _linear_in(model, scope + "/conv1d")
    , _linear_out(model, scope + "/conv1d_1") {
  }

  onmt::StorageView& operator()(const onmt::StorageView& queries,
                                const onmt::StorageView& queries_lengths,
                                onmt::StorageView* cached_keys = nullptr,
                                onmt::StorageView* cached_values = nullptr,
                                ssize_t step = 0) {
    const onmt::StorageView& normed_queries = _layer_norm(queries);
    const onmt::StorageView& fused_proj = _linear_in(normed_queries);

    std::vector<onmt::StorageView*> split_proj{&_queries_proj, &_keys_proj, &_values_proj};
    onmt::ops::Split(-1)(fused_proj, split_proj);
    onmt::StorageView values_lengths(queries_lengths);

    onmt::StorageView keys_proj;
    onmt::StorageView values_proj;

    if (step >= 0 && cached_keys != nullptr) {
      cache_proj(step, _keys_proj, *cached_keys);
      cache_proj(step, _values_proj, *cached_values);
      keys_proj.shallow_copy(*cached_keys);
      values_proj.shallow_copy(*cached_values);
      values_lengths.fill(static_cast<int32_t>(step + 1));
    } else {
      keys_proj.shallow_copy(_keys_proj);
      values_proj.shallow_copy(_values_proj);
    }

    const onmt::StorageView& attention_output = compute_attention(_queries_proj,
                                                                  keys_proj,
                                                                  values_proj,
                                                                  queries_lengths,
                                                                  values_lengths);

    onmt::StorageView& output = _linear_out(attention_output);
    onmt::compute::add(queries.data<float>(), output.data<float>(), queries.size());
    return output;
  }

  static void cache_proj(ssize_t step, const onmt::StorageView& proj, onmt::StorageView& cache) {
    assert(proj.rank() == 2);
    if (step == 0) {
      cache = proj;
      return;
    }
    assert(cache.rank() == 2);
    size_t batch_size = proj.dim(0);
    size_t depth = proj.dim(1);
    static onmt::StorageView tmp;
    tmp = cache;
    cache.grow(0, batch_size);
    const auto* src = tmp.data<float>();
    auto* dst = cache.data<float>();
    for (size_t i = 0; i < batch_size; ++i) {
      onmt::compute::copy(src, dst, step * depth);
      src += step * depth;
      dst += step * depth;
      onmt::compute::copy(proj.index<float>({i}), dst, depth);
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
  onmt::StorageView _keys_proj;
  onmt::StorageView _values_proj;

public:
  TransformerAttention(const onmt::Model& model,
                       const std::string& scope,
                       size_t num_heads)
    : MultiHeadAttention(model, scope, num_heads)
    , _linear_query(model, scope + "/conv1d")
    , _linear_memory(model, scope + "/conv1d_1")
    , _linear_out(model, scope + "/conv1d_2") {
  }

  onmt::StorageView& operator()(const onmt::StorageView& queries,
                                const onmt::StorageView& queries_lengths,
                                const onmt::StorageView& memory,
                                const onmt::StorageView& memory_lengths,
                                onmt::StorageView* cached_keys = nullptr,
                                onmt::StorageView* cached_values = nullptr,
                                ssize_t step = -1) {
    const onmt::StorageView& normed_queries = _layer_norm(queries);
    const onmt::StorageView& queries_proj = _linear_query(normed_queries);

    if (step > 0 && cached_keys != nullptr && !cached_keys->empty()) {
      _keys_proj.shallow_copy(*cached_keys);
      _values_proj.shallow_copy(*cached_values);
    } else {
      const onmt::StorageView& memory_proj = _linear_memory(memory);

      std::vector<onmt::StorageView*> split_proj{&_keys_proj, &_values_proj};
      onmt::ops::Split(-1)(memory_proj, split_proj);
      if (cached_keys != nullptr) {
        *cached_keys = _keys_proj;
        *cached_values = _values_proj;
      }
    }

    const onmt::StorageView& attention_output = compute_attention(queries_proj,
                                                                  _keys_proj,
                                                                  _values_proj,
                                                                  queries_lengths,
                                                                  memory_lengths);

    onmt::StorageView& output = _linear_out(attention_output);
    onmt::compute::add(queries.data<float>(), output.data<float>(), queries.size());
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

  onmt::StorageView& operator()(const onmt::StorageView& input,
                                const onmt::StorageView& lengths) {
    const auto& context = _self_attention(input, lengths);
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

  onmt::StorageView& operator()(size_t step,
                                const onmt::StorageView& input,
                                const onmt::StorageView& input_lengths,
                                const onmt::StorageView& memory,
                                const onmt::StorageView& memory_lengths,
                                onmt::StorageView& cached_self_attn_keys,
                                onmt::StorageView& cached_self_attn_values,
                                onmt::StorageView& cached_attn_keys,
                                onmt::StorageView& cached_attn_values) {
    const auto& encoded = _self_attention(
      input, input_lengths, &cached_self_attn_keys, &cached_self_attn_values, step);
    const auto& context = _encoder_attention(
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

  onmt::StorageView& operator()(const onmt::StorageView& ids,
                                const onmt::StorageView& lengths) {
    const auto& embeddings = _scaled_embeddings(ids);
    const auto& input = _position_encoder(embeddings, lengths);
    const auto* x = &input;
    for (auto& layer : _layers) {
      x = &layer(*x, lengths);
    }
    return _output_norm(*x);
  }
};

struct TransformerDecoderState {
  TransformerDecoderState()
    : memory_lengths(onmt::DataType::DT_INT32) {
  }
  onmt::StorageView memory;
  onmt::StorageView memory_lengths;
  std::vector<onmt::StorageView> cache;
};

static void remove_batch(onmt::StorageView& s,
                         const onmt::StorageView& lengths,
                         const std::vector<bool>& finished) {
  assert(s.rank() == 2);
  static onmt::StorageView tmp;
  tmp = s;
  size_t batch_size = lengths.dim(0);
  size_t depth = s.dim(1);
  const auto* src = tmp.data<float>();
  auto* dst = s.data<float>();
  size_t cum_length = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    const auto length = lengths.data<int32_t>()[i];
    const auto count = length * depth;
    if (!finished[i]) {
      onmt::compute::copy(src, dst, count);
      dst += count;
      cum_length += length;
    }
    src += count;
  }
  s.resize(0, cum_length);
}

static void remove_batch(onmt::StorageView& s, const std::vector<bool>& finished) {
  assert(s.rank() == 1);
  size_t write_index = 0;
  size_t read_index = 0;
  auto* s_buf = s.data<int32_t>();
  while (read_index < s.dim(0)) {
    if (!finished[read_index])
      s_buf[write_index++] = s_buf[read_index];
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

  void reset_state(const onmt::StorageView& memory,
                   const onmt::StorageView& memory_lengths) {
    _state.memory = memory;
    _state.memory_lengths = memory_lengths;
  }

  void prune_batch(size_t step, const std::vector<bool>& ids) {
    size_t batch_size = _state.memory_lengths.dim(0);
    onmt::StorageView step_lengths({batch_size}, static_cast<int32_t>(step + 1));
    for (size_t l = 0; l < _layers.size(); ++l) {
      remove_batch(_state.cache[0 + l * 4], step_lengths, ids);
      remove_batch(_state.cache[1 + l * 4], step_lengths, ids);
      remove_batch(_state.cache[2 + l * 4], _state.memory_lengths, ids);
      remove_batch(_state.cache[3 + l * 4], _state.memory_lengths, ids);
    }
    remove_batch(_state.memory, _state.memory_lengths, ids);
    remove_batch(_state.memory_lengths, ids);
  }

  onmt::StorageView& operator()(size_t step, const onmt::StorageView& ids) {
    size_t batch_size = ids.dim(0);
    const auto& embeddings = _scaled_embeddings(ids);
    const auto& input = _position_encoder(embeddings, step);
    onmt::StorageView query_lengths({batch_size}, static_cast<int32_t>(1));

    const auto* x = &input;
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
    const auto& normed = _output_norm(*x);
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
  onmt::StorageView lengths({input_tokens.size()}, onmt::DataType::DT_INT32);
  size_t total_length = 0;
  for (size_t i = 0; i < input_tokens.size(); ++i) {
    const size_t length = input_tokens[i].size();
    lengths.data<int32_t>()[i] = length;
    total_length += length;
  }

  onmt::StorageView ids({total_length}, onmt::DataType::DT_INT32);
  size_t offset = 0;
  for (const auto& tokens : input_tokens) {
    for (const auto& token : tokens) {
      ids.data<int32_t>()[offset] = vocabulary.to_id(token);
      offset += 1;
    }
  }

  // for (auto id : ids)
  //   std::cout << id << std::endl;

  size_t batch_size = lengths.size();
  const auto& encoded = encoder(ids, lengths);

  decoder.reset_state(encoded, lengths);

  onmt::StorageView sample_from({batch_size}, static_cast<int32_t>(vocabulary.to_id("<s>")));
  onmt::StorageView probs({batch_size, vocabulary.size()});
  std::vector<std::vector<size_t> > sampled_ids(batch_size);
  std::vector<bool> finished(batch_size, false);
  std::vector<size_t> batch_offset(batch_size);
  for (size_t i = 0; i < batch_offset.size(); ++i)
    batch_offset[i] = i;
  size_t max_steps = 200;

  for (size_t step = 0; step < max_steps; ++step) {
    const auto& logits = decoder(step, sample_from);
    onmt::ops::SoftMax()(logits, probs);

    std::vector<bool> finished_batch(logits.dim(0), false);
    bool one_finished = false;
    for (size_t i = 0; i < logits.dim(0); ++i) {
      size_t best = onmt::compute::max_element(probs.index<float>({i}), vocabulary.size());
      size_t batch_id = batch_offset[i];
      if (best == 2) {
        finished[batch_id] = true;
        finished_batch[i] = true;
        one_finished = true;
      } else {
        sample_from.data<int32_t>()[i] = best;
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
