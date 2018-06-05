#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <cstring>
#include <unordered_map>
#include <chrono>

#include "opennmt/vocabulary.h"
#include "opennmt/encoder.h"
#include "opennmt/decoder.h"
#include "opennmt/ops/ops.h"

#include "model.h"

class ScaledEmbeddings
{
public:
  ScaledEmbeddings(const opennmt::Model& model, const std::string& scope)
    : _embeddings(model.get_variable(scope + "/w_embs"))
    , _gathered(_embeddings.dtype()) {
  }

  opennmt::StorageView& operator()(const opennmt::StorageView& ids) {
    if (_embeddings.dtype() == opennmt::DataType::DT_FLOAT) {
      _gather_op(_embeddings, ids, _output);
    } else {
      _gather_op(_embeddings, ids, _gathered);
      opennmt::ops::Unquantize(1000)(_gathered, _output);
    }
    const size_t embedding_size = _embeddings.dim(-1);
    opennmt::primitives::mul(static_cast<float>(sqrt(embedding_size)),
                             _output.data<float>(),
                             _output.size());
    return _output;
  }

private:
  opennmt::ops::Gather _gather_op;
  const opennmt::StorageView& _embeddings;
  opennmt::StorageView _gathered;
  opennmt::StorageView _output;
};

class PositionEncoder
{
public:
  opennmt::StorageView& operator()(const opennmt::StorageView& input,
                                   const opennmt::StorageView& lengths) {
    assert(input.rank() == 3);
    size_t depth = input.dim(-1);
    if (_cached_encodings.empty())
      precompute_position_encoding(_max_cached_time, depth);
    _output = input;
    for (size_t i = 0; i < lengths.dim(0); ++i) {
      const auto length = lengths.at<int32_t>(i);
      opennmt::primitives::add(_cached_encodings.data<float>(),
                               _output.index<float>({i}),
                               length * depth);
    }
    return _output;
  }

  opennmt::StorageView& operator()(const opennmt::StorageView& input, size_t index) {
    size_t depth = input.dim(-1);
    if (_cached_encodings.empty())
      precompute_position_encoding(_max_cached_time, depth);
    _output = input;
    for (size_t i = 0; i < input.dim(0); ++i) {
      opennmt::primitives::add(_cached_encodings.index<float>({index}),
                               _output.index<float>({i}),
                               depth);
    }
    return _output;
  }

private:
  size_t _max_cached_time = 500;
  opennmt::StorageView _cached_encodings;
  opennmt::StorageView _output;

  void precompute_position_encoding(size_t max_time, size_t depth) {
    float log_timescale_increment = log(10000) / (depth / 2 - 1);
    opennmt::StorageView timescales({depth / 2}, -log_timescale_increment);
    for (size_t i = 0; i < timescales.size(); ++i)
      timescales.data<float>()[i] = exp(timescales.data<float>()[i] * i);

    opennmt::StorageView scaled_time({max_time, depth / 2});
    for (size_t i = 0; i < scaled_time.dim(0); ++i) {
      for (size_t j = 0; j < scaled_time.dim(1); ++j) {
        *scaled_time.index<float>({i, j}) = (i + 1) * timescales.data<float>()[j];
      }
    }

    opennmt::StorageView sin_encoding;
    opennmt::StorageView cos_encoding;

    opennmt::ops::Sin()(scaled_time, sin_encoding);
    opennmt::ops::Cos()(scaled_time, cos_encoding);
    opennmt::ops::Concat(-1)({&sin_encoding, &cos_encoding}, _cached_encodings);
  }
};

class Dense
{
public:
  Dense(const opennmt::Model& model, const std::string& scope)
    : _gemm_op(1, 1, true, false, true)
    , _weight(model.get_variable(scope + "/kernel"))
    , _bias(model.get_variable(scope + "/bias")) {
  }

  opennmt::StorageView& operator()(const opennmt::StorageView& input) {
    _gemm_op(input, _weight, _bias, _output);
    return _output;
  }

private:
  opennmt::ops::Gemm _gemm_op;
  const opennmt::StorageView& _weight;
  const opennmt::StorageView& _bias;
  opennmt::StorageView _output;
};

class LayerNorm
{
public:
  LayerNorm(const opennmt::Model& model, const std::string& scope)
    : _beta(model.get_variable(scope + "/beta"))
    , _gamma(model.get_variable(scope + "/gamma")) {
  }

  opennmt::StorageView& operator()(const opennmt::StorageView& input) {
    _norm_op(_beta, _gamma, input, _output);
    return _output;
  }

private:
  opennmt::ops::LayerNorm _norm_op;
  const opennmt::StorageView& _beta;
  const opennmt::StorageView& _gamma;
  opennmt::StorageView _output;
};

class TransformerFeedForward
{
public:
  TransformerFeedForward(const opennmt::Model& model,
                         const std::string& scope)
    : _layer_norm(model, scope + "/LayerNorm")
    , _ff1(model, scope + "/conv1d")
    , _ff2(model, scope + "/conv1d_1") {
  }

  opennmt::StorageView& operator()(const opennmt::StorageView& input) {
    const opennmt::StorageView& normed = _layer_norm(input);
    opennmt::StorageView& inner = _ff1(normed);
    opennmt::ops::ReLU()(inner);
    opennmt::StorageView& outer = _ff2(inner);
    opennmt::primitives::add(input.data<float>(), outer.data<float>(), input.size());
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

  opennmt::StorageView& operator()(const opennmt::StorageView& queries,
                                   const opennmt::StorageView& keys,
                                   const opennmt::StorageView& values,
                                   const opennmt::StorageView& values_lengths) {
    assert(queries.rank() == 4);
    assert(keys.rank() == 4);
    assert(values.rank() == 4);

    size_t batch_size = queries.dim(0);
    size_t num_heads = queries.dim(1);
    size_t queries_time = queries.dim(2);
    size_t memory_time = keys.dim(2);

    opennmt::ops::MatMul(false, true)(queries, keys, _dot);

    if (batch_size > 1) {
      for (size_t b = 0; b < batch_size; ++b) {
        const size_t length = values_lengths.data<int32_t>()[b];
        if (length == memory_time)
          continue;
        for (size_t h = 0; h < num_heads; ++h) {
          for (size_t i = 0; i < queries_time; ++i) {
            auto* x = _dot.index<float>({b, h, i});
            opennmt::primitives::fill(x + length, std::numeric_limits<float>::lowest(), memory_time - length);
          }
        }
      }
    }

    opennmt::ops::SoftMax()(_dot, _attn);
    opennmt::ops::MatMul()(_attn, values, _dot);
    return _dot;
  }

private:
  opennmt::StorageView _dot;
  opennmt::StorageView _attn;
};

class MultiHeadAttention
{
public:
  MultiHeadAttention(const opennmt::Model& model,
                     const std::string& scope,
                     size_t num_heads)
    : _layer_norm(model, scope + "/LayerNorm")
    , _num_heads(num_heads)
    , _transpose_op({0, 2, 1, 3}) {
  }

  void split_heads(const opennmt::StorageView& x, opennmt::StorageView& y) {
    opennmt::StorageView z(const_cast<float*>(x.data<float>()),
                           {x.dim(0), x.dim(1), _num_heads, x.dim(2) / _num_heads});
    _transpose_op(z, y);
  }

  void combine_heads(const opennmt::StorageView& x, opennmt::StorageView& y) {
    _transpose_op(x, y);
    y.reshape({y.dim(0), y.dim(1), y.dim(-1) * _num_heads});
  }

  opennmt::StorageView& compute_attention(const opennmt::StorageView& queries,
                                          const opennmt::StorageView& keys,
                                          const opennmt::StorageView& values,
                                          const opennmt::StorageView& queries_lengths,
                                          const opennmt::StorageView& values_lengths) {
    split_heads(queries, _split_queries);
    split_heads(keys, _split_keys);
    split_heads(values, _split_values);

    const size_t dk = queries.dim(-1) / _num_heads;
    opennmt::primitives::mul(static_cast<float>(1.0 / sqrt(dk)),
                             _split_queries.data<float>(),
                             _split_queries.size());

    const opennmt::StorageView& context = _attention(_split_queries,
                                                     _split_keys,
                                                     _split_values,
                                                     values_lengths);

    combine_heads(context, _combined);
    return _combined;
  }

protected:
  LayerNorm _layer_norm;

private:
  size_t _num_heads;
  DotProductAttention _attention;
  opennmt::ops::Transpose _transpose_op;
  opennmt::StorageView _split_queries;
  opennmt::StorageView _split_keys;
  opennmt::StorageView _split_values;
  opennmt::StorageView _combined;
};

class TransformerSelfAttention : public MultiHeadAttention
{
private:
  Dense _linear_in;
  Dense _linear_out;
  opennmt::StorageView _queries_proj;
  opennmt::StorageView _keys_proj;
  opennmt::StorageView _values_proj;

public:
  TransformerSelfAttention(const opennmt::Model& model,
                           const std::string& scope,
                           size_t num_heads)
    : MultiHeadAttention(model, scope, num_heads)
    , _linear_in(model, scope + "/conv1d")
    , _linear_out(model, scope + "/conv1d_1") {
  }

  opennmt::StorageView& operator()(const opennmt::StorageView& queries,
                                   const opennmt::StorageView& queries_lengths,
                                   opennmt::StorageView* cached_keys = nullptr,
                                   opennmt::StorageView* cached_values = nullptr,
                                   ssize_t step = 0) {
    const opennmt::StorageView& normed_queries = _layer_norm(queries);
    const opennmt::StorageView& fused_proj = _linear_in(normed_queries);
    opennmt::ops::Split(-1)(fused_proj, _queries_proj, _keys_proj, _values_proj);

    opennmt::StorageView values_lengths(queries_lengths);
    opennmt::StorageView keys_proj;
    opennmt::StorageView values_proj;

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

    const opennmt::StorageView& attention_output = compute_attention(_queries_proj,
                                                                     keys_proj,
                                                                     values_proj,
                                                                     queries_lengths,
                                                                     values_lengths);

    opennmt::StorageView& output = _linear_out(attention_output);
    opennmt::primitives::add(queries.data<float>(), output.data<float>(), queries.size());
    return output;
  }

  static void cache_proj(ssize_t step, opennmt::StorageView& proj, opennmt::StorageView& cache) {
    if (step == 0) {
      cache = proj;
    } else {
      static opennmt::StorageView tmp;
      tmp = cache;
      opennmt::ops::Concat(1)({&tmp, &proj}, cache);
    }
  }

};

class TransformerAttention : public MultiHeadAttention
{
private:
  Dense _linear_query;
  Dense _linear_memory;
  Dense _linear_out;
  opennmt::StorageView _keys_proj;
  opennmt::StorageView _values_proj;

public:
  TransformerAttention(const opennmt::Model& model,
                       const std::string& scope,
                       size_t num_heads)
    : MultiHeadAttention(model, scope, num_heads)
    , _linear_query(model, scope + "/conv1d")
    , _linear_memory(model, scope + "/conv1d_1")
    , _linear_out(model, scope + "/conv1d_2") {
  }

  opennmt::StorageView& operator()(const opennmt::StorageView& queries,
                                   const opennmt::StorageView& queries_lengths,
                                   const opennmt::StorageView& memory,
                                   const opennmt::StorageView& memory_lengths,
                                   opennmt::StorageView* cached_keys = nullptr,
                                   opennmt::StorageView* cached_values = nullptr,
                                   ssize_t step = -1) {
    const opennmt::StorageView& normed_queries = _layer_norm(queries);
    const opennmt::StorageView& queries_proj = _linear_query(normed_queries);

    if (step > 0 && cached_keys != nullptr && !cached_keys->empty()) {
      _keys_proj.shallow_copy(*cached_keys);
      _values_proj.shallow_copy(*cached_values);
    } else {
      const opennmt::StorageView& memory_proj = _linear_memory(memory);
      opennmt::ops::Split(-1)(memory_proj, _keys_proj, _values_proj);
      if (cached_keys != nullptr) {
        *cached_keys = _keys_proj;
        *cached_values = _values_proj;
      }
    }

    const opennmt::StorageView& attention_output = compute_attention(queries_proj,
                                                                     _keys_proj,
                                                                     _values_proj,
                                                                     queries_lengths,
                                                                     memory_lengths);

    opennmt::StorageView& output = _linear_out(attention_output);
    opennmt::primitives::add(queries.data<float>(), output.data<float>(), queries.size());
    return output;
  }
};

class TransformerEncoderLayer
{
public:
  TransformerEncoderLayer(const opennmt::Model& model,
                          const std::string& scope)
    : _self_attention(model, scope + "/multi_head", 8)
    , _ff(model, scope + "/ffn") {
  }

  opennmt::StorageView& operator()(const opennmt::StorageView& input,
                                   const opennmt::StorageView& lengths) {
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
  TransformerDecoderLayer(const opennmt::Model& model,
                          const std::string& scope)
    : _self_attention(model, scope + "/masked_multi_head", 8)
    , _encoder_attention(model, scope + "/multi_head", 8)
    , _ff(model, scope + "/ffn") {
  }

  opennmt::StorageView& operator()(size_t step,
                                   const opennmt::StorageView& input,
                                   const opennmt::StorageView& input_lengths,
                                   const opennmt::StorageView& memory,
                                   const opennmt::StorageView& memory_lengths,
                                   opennmt::StorageView& cached_self_attn_keys,
                                   opennmt::StorageView& cached_self_attn_values,
                                   opennmt::StorageView& cached_attn_keys,
                                   opennmt::StorageView& cached_attn_values) {
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

class TransformerEncoder : public opennmt::Encoder
{
public:
  TransformerEncoder(const opennmt::Model& model, const std::string& scope)
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

  opennmt::StorageView& encode(const opennmt::StorageView& ids,
                               const opennmt::StorageView& lengths) override {
    const auto& embeddings = _scaled_embeddings(ids);
    const auto& input = _position_encoder(embeddings, lengths);
    const auto* x = &input;
    for (auto& layer : _layers) {
      x = &layer(*x, lengths);
    }
    return _output_norm(*x);
  }

private:
  ScaledEmbeddings _scaled_embeddings;
  PositionEncoder _position_encoder;
  LayerNorm _output_norm;
  std::vector<TransformerEncoderLayer> _layers;
};

class TransformerDecoderState : public opennmt::DecoderState {
public:
  TransformerDecoderState(size_t num_layers)
    : DecoderState() {
    for (size_t i = 0; i < num_layers; ++i) {
      add("self_keys_" + std::to_string(i));
      add("self_values_" + std::to_string(i));
      add("memory_keys_" + std::to_string(i));
      add("memory_values_" + std::to_string(i));
    }
  }

  void reset(const opennmt::StorageView& memory,
             const opennmt::StorageView& memory_lengths) override {
    opennmt::DecoderState::reset(memory, memory_lengths);
    for (auto& pair : _states) {
      if (pair.first != "memory" and pair.first != "memory_lengths")
        pair.second.clear();
    }
  }
};

class TransformerDecoder : public opennmt::Decoder
{
public:
  TransformerDecoder(const opennmt::Model& model, const std::string& scope)
    : _scaled_embeddings(model, scope)
    , _position_encoder()
    , _output_norm(model, scope + "/LayerNorm")
    , _proj(model, scope + "/dense") {
    for (size_t l = 0;; ++l) {
      try {
        _layers.emplace_back(model, scope + "/layer_" + std::to_string(l));
      } catch (std::exception&) {
        break;
      }
    }
    _state.reset(new TransformerDecoderState(_layers.size()));
  }

  opennmt::StorageView& logits(size_t step, const opennmt::StorageView& ids) override {
    size_t batch_size = ids.dim(0);
    const auto& embeddings = _scaled_embeddings(ids);
    const auto& input = _position_encoder(embeddings, step);
    opennmt::StorageView query_lengths({batch_size}, static_cast<int32_t>(1));

    const auto* x = &input;
    for (size_t l = 0; l < _layers.size(); ++l) {
      x = &_layers[l](step,
                      *x,
                      query_lengths,
                      _state->get("memory"),
                      _state->get("memory_lengths"),
                      _state->get("self_keys_" + std::to_string(l)),
                      _state->get("self_values_" + std::to_string(l)),
                      _state->get("memory_keys_" + std::to_string(l)),
                      _state->get("memory_values_" + std::to_string(l)));
    }
    const auto& normed = _output_norm(*x);
    return _proj(normed);
  }

private:
  ScaledEmbeddings _scaled_embeddings;
  PositionEncoder _position_encoder;
  LayerNorm _output_norm;
  std::vector<TransformerDecoderLayer> _layers;
  Dense _proj;
};

void translate(const std::vector<std::vector<std::string> >& input_tokens,
               const opennmt::Vocabulary& vocabulary,
               opennmt::Encoder& encoder,
               opennmt::Decoder& decoder,
               size_t beam_size) {
  size_t batch_size = input_tokens.size();
  size_t max_length = 0;
  opennmt::StorageView lengths({batch_size}, opennmt::DataType::DT_INT32);
  for (size_t i = 0; i < batch_size; ++i) {
    const size_t length = input_tokens[i].size();
    lengths.at<int32_t>(i) = length;
    max_length = std::max(max_length, length);
  }

  opennmt::StorageView ids({batch_size, max_length}, opennmt::DataType::DT_INT32);
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t t = 0; t < input_tokens[i].size(); ++t) {
      ids.at<int32_t>({i, t}) = vocabulary.to_id(input_tokens[i][t]);
    }
  }

  const auto& encoded = encoder.encode(ids, lengths);

  opennmt::StorageView sample_from({batch_size, 1}, static_cast<int32_t>(vocabulary.to_id("<s>")));
  std::vector<std::vector<size_t>> sampled_ids;

  decoder.get_state().reset(encoded, lengths);
  if (beam_size == 1)
    opennmt::greedy_decoding(decoder, sample_from, 2, vocabulary.size(), 200, sampled_ids);
  else
    opennmt::beam_search(decoder, sample_from, 2, 5, 0.6, vocabulary.size(), 200, sampled_ids);

  for (size_t i = 0; i < batch_size; ++i) {
    for (auto id : sampled_ids[i]) {
      std::cout << " " << vocabulary.to_token(id);
    }
    std::cout << std::endl;
  }
}

int main(int argc, char* argv[]) {
  opennmt::Model model("/home/klein/dev/ctransformer/model.bin");
  opennmt::Vocabulary vocabulary("/home/klein/data/wmt-ende/wmtende.vocab");

  TransformerEncoder encoder(model, "transformer/encoder");
  TransformerDecoder decoder(model, "transformer/decoder");

  std::ifstream text_file("/home/klein/data/wmt-ende/valid.en");
  std::vector<std::vector<std::string> > input_tokens;
  std::string line;

  size_t max_batch_size = argc > 1 ? std::stoi(argv[1]) : 1;
  size_t beam_size = argc > 2 ? std::stoi(argv[2]) : 1;
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
      translate(input_tokens, vocabulary, encoder, decoder, beam_size);
      input_tokens.clear();
    }
  }

  if (!input_tokens.empty()) {
    translate(input_tokens, vocabulary, encoder, decoder, beam_size);
  }

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  std::cerr << static_cast<double>(num_tokens) / static_cast<double>(duration / 1000) << std::endl;
  return 0;
}
