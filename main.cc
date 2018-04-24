#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <cstring>
#include <unordered_map>

#include "model.h"
#include "vocabulary.h"
#include "routines.h"
#include "storage_view.h"

void pad_sequences(const StorageView<float>& flattened,
                   const StorageView<size_t>& lengths,
                   StorageView<float>& padded) {
  size_t batch_size = lengths.dim(0);
  size_t max_length = *std::max_element(lengths.data(), lengths.data() + lengths.size());
  size_t depth = flattened.dim(1);
  padded.resize({batch_size, max_length, depth});
  const float* src = flattened.data();
  float* dst = padded.data();
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

class Node
{
public:
  virtual ~Node() = default;
protected:
  StorageView<float> _output;
};

class ScaledEmbeddings : public Node
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
  const StorageView<float>& _weight;
};

class PositionEncoder : public Node
{
public:
  StorageView<float>& operator()(const StorageView<float>& input, size_t index) {
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
  size_t _max_cached_time = 300;
  StorageView<float> _cached_encodings;

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

class Dense : public Node
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
    size_t input_depth = shape.back();
    size_t output_depth_ = output_depth();
    _output.resize({batch_size, output_depth_});
    linear(input.data(), _weight.data(), _bias.data(),
           batch_size, input_depth, output_depth_, _output.data());
    return _output;
  }

  size_t output_depth() const {
    return _weight.shape().back();
  }

private:
  const StorageView<float>& _weight;
  const StorageView<float>& _bias;
};

class LayerNorm : public Node
{
public:
  LayerNorm(const Model& model, const std::string& scope)
    : _beta(model.get_variable(scope + "/beta"))
    , _gamma(model.get_variable(scope + "/gamma")) {
  }

  StorageView<float>& operator()(const StorageView<float>& input) {
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
  const StorageView<float>& _beta;
  const StorageView<float>& _gamma;
  StorageView<float> _tmp;
};

class TransformerFeedForward : public Node
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

class DotProductAttention : public Node
{
public:

  StorageView<float>& operator()(const StorageView<float>& queries,
                                 const StorageView<float>& keys,
                                 const StorageView<float>& values,
                                 const StorageView<size_t>* values_lengths) {
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

    if (batch_size > 1 && values_lengths != nullptr) {
      for (size_t b = 0; b < batch_size; ++b) {
        const size_t length = (*values_lengths)[b];
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

    _output.resize_as(queries);
    batch_mat_mul(_attn.data(), values.data(),
                  CblasNoTrans, CblasNoTrans,
                  batch_size * num_heads, queries_time, depth, memory_time,
                  _output.data());
    return _output;
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
    : _num_heads(num_heads)
    , _layer_norm(model, scope + "/LayerNorm") {
    for (size_t i = 0;; ++i) {
      std::string conv_scope = scope + "/conv1d";
      if (i > 0)
        conv_scope += "_" + std::to_string(i);

      try {
        _projections.emplace_back(model, conv_scope);
      } catch(std::exception&) {
        break;
      }
    }

    _depth = _projections.back().output_depth();
  }

  StorageView<float>& operator()(const StorageView<float>& queries,
                                 const StorageView<size_t>& queries_length,
                                 const StorageView<float>* memory = nullptr,
                                 const StorageView<size_t>* memory_length = nullptr,
                                 int step = -1) {
    if (memory_length == nullptr)
      memory_length = &queries_length;

    size_t batch_size = queries_length.dim(0);
    size_t queries_time = *std::max_element(queries_length.data(),
                                            queries_length.data() + queries_length.size());
    size_t memory_time = *std::max_element(memory_length->data(),
                                           memory_length->data() + memory_length->size());

    const StorageView<float>& normed_queries = _layer_norm(queries);
    StorageView<float> queries_proj;
    StorageView<float> keys_proj;
    StorageView<float> values_proj;

    if (memory == nullptr) {
      const StorageView<float>& fused_proj = _projections[0](normed_queries);

      _splits.resize_as(fused_proj);
      std::vector<float*> splits = split_in_depth(fused_proj.data(),
                                                  fused_proj.dim(0), fused_proj.dim(1),
                                                  3, _splits.data());

      queries_proj = StorageView<float>(splits[0], {fused_proj.dim(0), _depth});
      keys_proj = StorageView<float>(splits[1], {fused_proj.dim(0), _depth});
      values_proj = StorageView<float>(splits[2], {fused_proj.dim(0), _depth});

      // TODO
      if (step >= 0) {
        keys_proj = push_proj(keys_proj, _keys_accu, step);
        values_proj = push_proj(values_proj, _values_accu, step);
      }
    } else {
      StorageView<float>& proj = _projections[0](normed_queries);
      queries_proj = StorageView<float>(proj.data(), {proj.dim(0), _depth});
      if (step > 0 && !_cached_memory_keys.empty()) {
        keys_proj = _cached_memory_keys;
        values_proj = _cached_memory_values;
      } else {
        const StorageView<float>& fused_proj = _projections[1](*memory);
        _splits.resize_as(fused_proj);
        std::vector<float*> splits = split_in_depth(fused_proj.data(),
                                                    fused_proj.dim(0), fused_proj.dim(1),
                                                    2, _splits.data());
        keys_proj = StorageView<float>(splits[0], {fused_proj.dim(0), _depth});
        values_proj = StorageView<float>(splits[1], {fused_proj.dim(0), _depth});
        _cached_memory_keys = keys_proj;
        _cached_memory_values = values_proj;
      }
    }

    size_t dk = _depth / _num_heads;
    array_mul(1.0 / sqrt(dk), queries_proj.data(), queries_proj.size());

    pad_sequences(queries_proj, queries_length, _padded_queries);
    pad_sequences(keys_proj, *memory_length, _padded_keys);
    pad_sequences(values_proj, *memory_length, _padded_values);

    _split_queries.resize({batch_size, _num_heads, queries_time, dk});
    _split_keys.resize({batch_size, _num_heads, memory_time, dk});
    _split_values.resize_as(_split_keys);

    swap_middle_dims(_padded_queries.data(),
                     batch_size, queries_time, _num_heads, dk, _split_queries.data());
    swap_middle_dims(_padded_keys.data(),
                     batch_size, memory_time, _num_heads, dk, _split_keys.data());
    swap_middle_dims(_padded_values.data(),
                     batch_size, memory_time, _num_heads, dk, _split_values.data());

    StorageView<float>& context = _attention(_split_queries,
                                             _split_keys,
                                             _split_values,
                                             memory_length);

    StorageView<float>& combined = _padded_queries;
    StorageView<float>& combined_pruned = queries_proj;
    swap_middle_dims(context.data(), batch_size, _num_heads, queries_time, dk, combined.data());
    unpad_sequences(combined.data(), queries_length.data(),
                    batch_size, queries_time, _depth, combined_pruned.data());

    StorageView<float>& output = _projections.back()(combined_pruned);
    array_add(queries.data(), output.data(), queries.size());
    return output;
  }

  static StorageView<float>& push_proj(const StorageView<float>& proj,
                                       StorageView<float>& accu,
                                       size_t step) {
    if (step == 0) {
      accu = proj;
    } else {
      static StorageView<float> tmp;
      tmp = accu;
      size_t batch_size = proj.dim(0);
      size_t depth = proj.dim(1);
      accu.resize({batch_size * step + 1, depth});
      const float* src = tmp.data();
      float* dst = accu.data();
      for (size_t i = 0; i < batch_size; ++i) {
        array_copy(src, dst, step * depth);
        src += step * depth;
        dst += step * depth;
        array_copy(proj.index({i}), dst, depth);
        dst += depth;
      }
    }
    return accu;
  }

private:
  size_t _num_heads;
  size_t _depth;
  LayerNorm _layer_norm;
  DotProductAttention _attention;
  std::vector<Dense> _projections;

  StorageView<float> _cached_memory_keys;
  StorageView<float> _cached_memory_values;
  StorageView<float> _splits;
  StorageView<float> _keys_accu;
  StorageView<float> _values_accu;
  StorageView<float> _padded_queries;
  StorageView<float> _padded_keys;
  StorageView<float> _padded_values;
  StorageView<float> _split_queries;
  StorageView<float> _split_keys;
  StorageView<float> _split_values;
};

class TransformerEncoderLayer : public Node
{
public:
  TransformerEncoderLayer(const Model& model,
                          const std::string& scope)
    : _multi_head_attention(model, scope + "/multi_head", 8)
    , _ff(model, scope + "/ffn") {
  }

  StorageView<float>& operator()(const StorageView<float>& input,
                                 const StorageView<size_t>& lengths) {
    const StorageView<float>& context = _multi_head_attention(input, lengths);
    return _ff(context);
  }

private:
  MultiHeadAttention _multi_head_attention;
  TransformerFeedForward _ff;
};

class TransformerDecoderLayer : public Node
{
public:
  TransformerDecoderLayer(const Model& model,
                          const std::string& scope)
    : _masked_multi_head_attention(model, scope + "/masked_multi_head", 8)
    , _multi_head_attention(model, scope + "/multi_head", 8)
    , _ff(model, scope + "/ffn") {
  }

  StorageView<float>& operator()(size_t step,
                                 const StorageView<float>& input,
                                 const StorageView<float>& memory,
                                 const StorageView<size_t>& lengths,
                                 const StorageView<size_t>& history_lengths,
                                 const StorageView<size_t>& memory_lengths) {
    const StorageView<float>& encoded = _masked_multi_head_attention(
      input, lengths, nullptr, &history_lengths, step);
    const StorageView<float>& context = _multi_head_attention(
      encoded, lengths, &memory, &memory_lengths, step);
    return _ff(context);
  }

private:
  MultiHeadAttention _masked_multi_head_attention;
  MultiHeadAttention _multi_head_attention;
  TransformerFeedForward _ff;
};

template <typename TransformerLayer>
class TransformerStack : public Node
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
  std::vector<TransformerLayer> _layers;
  LayerNorm _output_norm;
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

class TransformerDecoder : public TransformerStack<TransformerDecoderLayer>
{
public:
  TransformerDecoder(const Model& model, const std::string& scope)
    : TransformerStack(model, scope)
    , _proj(model, scope + "/dense") {
  }

  StorageView<float>& operator()(size_t step,
                                 const StorageView<size_t>& ids,
                                 const StorageView<float>& memory,
                                 const StorageView<size_t>& memory_lengths) {
    size_t batch_size = ids.dim(0);
    const StorageView<float>& embeddings = _scaled_embeddings(ids);
    const StorageView<float>& input = _position_encoder(embeddings, step);
    StorageView<size_t> query_lengths({batch_size}, 1);
    StorageView<size_t> history_lengths({batch_size}, step + 1);

    const StorageView<float>* x = &input;
    for (auto& layer : _layers) {
      x = &layer(step, *x, memory, query_lengths, history_lengths, memory_lengths);
    }
    const StorageView<float>& normed = _output_norm(*x);
    return _proj(normed);
  }

private:
  Dense _proj;
  //DecoderState _state;
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

  StorageView<size_t> sample_from({batch_size}, vocabulary.to_id("<s>"));
  StorageView<float> probs({batch_size, vocabulary.size()});
  std::vector<std::vector<size_t> > sampled_ids(batch_size);
  std::vector<bool> finished(batch_size, false);
  bool all_finished = false;
  size_t max_steps = 200;

  for (size_t step = 0; step < max_steps && !all_finished; ++step) {
    StorageView<float>& logits = decoder(step, sample_from, encoded, lengths);
    softmax(logits.data(), batch_size, vocabulary.size(), probs.data());
    all_finished = true;
    for (size_t i = 0; i < batch_size; ++i) {
      size_t best = array_max_element(probs.index({i}), vocabulary.size());
      sample_from[i] = best;
      if (best == 2)
        finished[i] = true;
      else {
        all_finished = false;
        sampled_ids[i].push_back(best);
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

int main() {
  vmlSetMode(VML_EP);

  Model model("/home/klein/dev/ctransformer/model.bin");
  Vocabulary vocabulary("/home/klein/data/wmt-ende/wmtende.vocab");

  TransformerEncoder encoder(model, "transformer/encoder");
  TransformerDecoder decoder(model, "transformer/decoder");

  std::ifstream text_file("/home/klein/data/wmt-ende/valid.en");
  std::vector<std::vector<std::string> > input_tokens;
  std::string line;
  size_t max_batch_size = 1;
  size_t max_iter = 1000;
  size_t iter = 0;

  while (std::getline(text_file, line)) {
    std::cout << "  INPUT: " << line << std::endl;
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

    if (input_tokens.size() == max_batch_size) {
      translate(input_tokens, vocabulary, encoder, decoder);
      input_tokens.clear();
      iter += 1;
      if (iter >= max_iter)
        break;
    }
  }

  if (!input_tokens.empty()) {
    translate(input_tokens, vocabulary, encoder, decoder);
  }

  return 0;
}
