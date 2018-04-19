#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <cstring>
#include <unordered_map>

// #include <Eigen/Eigen>
// #include <unsupported/Eigen/CXX11/Tensor>
#include <mkl.h>
// #include <mkldnn.hpp>

#include "model.h"
#include "vocabulary.h"
#include "routines.h"

// using Tensor2D = Eigen::Tensor<float, 2, Eigen::RowMajor>;
// using Tensor3D = Eigen::Tensor<float, 3, Eigen::RowMajor>;
// using Tensor4D = Eigen::Tensor<float, 4, Eigen::RowMajor>;

template <typename T>
class ReusableVector
{
public:
  ReusableVector() {
  }
  ReusableVector(size_t size) {
    maybe_resize(size);
  }
  ~ReusableVector() {
    free(_buffer);
  }

  T* data() {
    return reinterpret_cast<T*>(_buffer);
  }

  size_t size() const {
    return _size;
  }

  void maybe_resize(size_t size, bool keep_content = false) {
    if (size > _alloc_size) {
      void* new_buffer = malloc(size * sizeof (T));
      if (keep_content)
        memcpy(new_buffer, _buffer, _size * sizeof (T));
      free(_buffer);
      _buffer = new_buffer;
      _alloc_size = size;
    }
    _size = size;
  }

private:
  void* _buffer = nullptr;
  size_t _size = 0;
  size_t _alloc_size = 0;
};

class Node
{
public:
  virtual ~Node() = default;
protected:
  float* output_buffer(size_t size) {
    _output.maybe_resize(size);
    return _output.data();
  }
private:
  ReusableVector<float> _output;
};

class ScaledEmbeddings : public Node
{
public:
  ScaledEmbeddings(const Model& model, const std::string& scope) {
    const Variable& weight = model.get_variable(scope + "/w_embs");
    _weight = weight.data();
    _size = weight.dim()[1];
  }

  float* operator()(const unsigned int* ids, unsigned int batch_size, float* output = nullptr) {
    unsigned int output_size = batch_size * _size;
    if (output == nullptr)
      output = output_buffer(output_size);
    gather(ids, _weight, batch_size, _size, output);
    array_mul(sqrt(_size), output, output_size);
    return output;
  }

  unsigned int output_depth() const {
    return _size;
  }

private:
  const float* _weight;
  unsigned int _size;
};

class PositionEncoder : public Node
{
public:
  PositionEncoder(unsigned int depth)
    : _depth(depth) {
  }

  float* operator()(const float* input,
                    unsigned int index,
                    unsigned int batch_size,
                    float* output = nullptr) {
    if (_cached_encodings.empty())
      precompute_position_encoding(300);
    if (output == nullptr)
      output = output_buffer(batch_size * _depth);
    array_copy(input, output, batch_size * _depth);
    const float* x = _cached_encodings.data() + (index * _depth);
    for (unsigned int i = 0; i < batch_size; ++i) {
      float* y = output + (i * _depth);
      array_add(x, y, _depth);
    }
    return output;
  }

  float* operator()(const float* input,
                    const unsigned int* lengths,
                    unsigned int batch_size,
                    unsigned int total_length,
                    float* output = nullptr) {
    if (_cached_encodings.empty())
      precompute_position_encoding(300);
    if (output == nullptr)
      output = output_buffer(total_length * _depth);
    array_copy(input, output, total_length * _depth);
    const float* x = _cached_encodings.data();
    unsigned int offset = 0;
    for (unsigned int i = 0; i < batch_size; ++i) {
      const unsigned int length = lengths[i];
      float* y = output + (offset * _depth);
      array_add(x, y, length * _depth);
      offset += length;
    }
    return output;
  }

private:
  unsigned int _depth;
  std::vector<float> _cached_encodings;

  void precompute_position_encoding(unsigned int max_time) {
    float log_timescale_increment = log(10000) / (_depth / 2 - 1);
    std::vector<float> timescales(_depth / 2, -log_timescale_increment);
    for (unsigned int i = 0; i < timescales.size(); ++i)
      timescales[i] = exp(timescales[i] * i);

    std::vector<float> scaled_time((_depth / 2) * max_time);
    for (unsigned int i = 0; i < max_time; ++i) {
      for (unsigned int j = 0; j < _depth / 2; ++j) {
        scaled_time[j + (i * _depth / 2)] = (i + 1) * timescales[j];
      }
    }

    std::vector<float> sin_encoding(scaled_time.size());
    std::vector<float> cos_encoding(scaled_time.size());

    vsSin(scaled_time.size(), scaled_time.data(), sin_encoding.data());
    vsCos(scaled_time.size(), scaled_time.data(), cos_encoding.data());

    _cached_encodings.resize(sin_encoding.size() + cos_encoding.size());
    concat_in_depth({sin_encoding.data(), cos_encoding.data()},
                    {_depth / 2, _depth / 2}, max_time,
                    _cached_encodings.data());
  }
};

class Dense : public Node
{
public:
  Dense(const Model& model, const std::string& scope) {
    const Variable& weight = model.get_variable(scope + "/kernel");
    const Variable& bias = model.get_variable(scope + "/bias");
    _weight = weight.data();
    _bias = bias.data();
    _input_depth = weight.dim()[weight.rank() - 2];
    _output_depth = weight.dim()[weight.rank() - 1];
  }

  float* operator()(const float* input,
                    unsigned int batch_size,
                    float* output = nullptr) {
    if (output == nullptr)
      output = output_buffer(batch_size * _output_depth);
    linear(input, _weight, _bias, batch_size, _input_depth, _output_depth, output);
    return output;
  }

  unsigned int output_depth() const {
    return _output_depth;
  }

private:
  const float* _weight;
  const float* _bias;
  unsigned int _input_depth;
  unsigned int _output_depth;
};

class LayerNorm : public Node
{
public:
  LayerNorm(const Model& model, const std::string& scope) {
    const Variable& beta = model.get_variable(scope + "/beta");
    const Variable& gamma = model.get_variable(scope + "/gamma");
    _beta = beta.data();
    _gamma = gamma.data();
    _depth = beta.dim()[0];
  }

  float* operator()(const float* input,
                    unsigned int batch_size,
                    float* output = nullptr) {
    if (output == nullptr)
      output = output_buffer(batch_size * _depth);
    _tmp.maybe_resize(_depth);
    for (unsigned int i = 0; i < batch_size; ++i) {
      const float* x = input + (i * _depth);
      float* y = output + (i * _depth);
      float mean = array_mean(x, _depth);
      array_copy(x, y, _depth);
      array_sub(mean, y, _depth); // y is now centered
      array_pow(y, _tmp.data(), 2, _depth);
      float variance = array_mean(_tmp.data(), _depth);
      array_mul(1.0 / sqrt(variance + EPSILON), y, _depth); // y is now centered and normalized.
      array_mul(_gamma, y, _depth);
      array_add(_beta, y, _depth);
    }
    return output;
  }

private:
  const float* _beta;
  const float* _gamma;
  unsigned int _depth;
  ReusableVector<float> _tmp;
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

  float* operator()(const float* input,
                    unsigned int batch_size,
                    float* output = nullptr) {
    float* normed = _layer_norm(input, batch_size);
    float* inner = _ff1(normed, batch_size);

    relu(inner, batch_size * _ff1.output_depth());

    // auto memory_desc = mkldnn::memory::desc({batch_size, _ff1.output_depth()},
    //                                         mkldnn::memory::data_type::f32,
    //                                         mkldnn::memory::format::nc);
    // auto relu_desc = mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward_inference,
    //                                                memory_desc,
    //                                                0);
    // auto memory_primitive_desc = mkldnn::memory::primitive_desc(memory_desc, _engine);
    // auto memory = mkldnn::memory(memory_primitive_desc, inner);

    // std::vector<mkldnn::primitive> primitives;
    // primitives.push_back(mkldnn::eltwise_forward(
    //                        mkldnn::eltwise_forward::primitive_desc(relu_desc, _engine),
    //                        mkldnn::primitive::at(memory),
    //                        memory));
    // mkldnn::stream(mkldnn::stream::kind::eager).submit(primitives).wait();

    output = _ff2(inner, batch_size, output);
    array_add(input, output, batch_size * _ff2.output_depth());
    return output;
  }

private:
  LayerNorm _layer_norm;
  Dense _ff1;
  Dense _ff2;
};

class DotProductAttention : public Node
{
public:

  float* operator()(const float* queries,
                    const float* keys,
                    const float* values,
                    unsigned int batch_size,
                    unsigned int queries_time,
                    unsigned int keys_time,
                    unsigned int depth,
                    float* output = nullptr) {
    _dot.maybe_resize(batch_size * queries_time * keys_time);
    batch_mat_mul(queries, keys,
                  CblasNoTrans, CblasTrans,
                  batch_size, queries_time, keys_time, depth,
                  _dot.data());
    _attn.maybe_resize(batch_size * queries_time * keys_time);
    softmax(_dot.data(), batch_size * queries_time, keys_time, _attn.data());
    if (output == nullptr)
      output = output_buffer(batch_size * queries_time * depth);
    batch_mat_mul(_attn.data(), values,
                  CblasNoTrans, CblasNoTrans,
                  batch_size, queries_time, depth, keys_time,
                  output);
    return output;
  }

private:
  ReusableVector<float> _dot;
  ReusableVector<float> _attn;
};

class MultiHeadAttention
{
public:
  MultiHeadAttention(const Model& model,
                     const std::string& scope,
                     unsigned int num_heads)
    : _num_heads(num_heads)
    , _layer_norm(model, scope + "/LayerNorm") {
    for (unsigned int i = 0;; ++i) {
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

  float* operator()(const float* queries,
                    const float* memory,
                    const unsigned int* queries_length,
                    const unsigned int* memory_length,
                    unsigned int batch_size,
                    int step = -1,
                    float* output = nullptr) {
    if (memory_length == nullptr)
      memory_length = queries_length;
    unsigned int cum_queries_length = 0;
    unsigned int cum_memory_length = 0;
    unsigned int queries_time = 0;
    unsigned int memory_time = 0;
    for (unsigned int i = 0; i < batch_size; ++i) {
      cum_queries_length += queries_length[i];
      queries_time = std::max(queries_time, queries_length[i]);
      cum_memory_length += memory_length[i];
      memory_time = std::max(memory_time, memory_length[i]);
    }

    float* normed_queries = _layer_norm(queries, cum_queries_length);
    float* queries_proj;
    const float* keys_proj;
    const float* values_proj;

    if (memory == nullptr) {
      float* fused_proj = _projections[0](normed_queries, cum_queries_length);
      unsigned int fused_depth = _projections[0].output_depth();
      _splits.maybe_resize(cum_queries_length * fused_depth);
      std::vector<float*> splits = split_in_depth(fused_proj, cum_queries_length, fused_depth,
                                                  3, _splits.data());
      queries_proj = splits[0];
      keys_proj = splits[1];
      values_proj = splits[2];
      if (step >= 0) {
        _keys_accu.maybe_resize((step + 1) * batch_size * _depth, true);
        _values_accu.maybe_resize((step + 1) * batch_size * _depth, true);
        array_copy(keys_proj, _keys_accu.data() + step * batch_size * _depth, batch_size * _depth);
        array_copy(values_proj, _values_accu.data() + step * batch_size * _depth, batch_size * _depth);
        keys_proj = _keys_accu.data();
        values_proj = _values_accu.data();
      }
    } else {
      queries_proj = _projections[0](normed_queries, cum_queries_length);
      if (step > 0 && _cached_memory_keys != nullptr) {
        keys_proj = _cached_memory_keys;
        values_proj = _cached_memory_values;
      } else {
        unsigned int fused_depth = _projections[1].output_depth();
        _splits.maybe_resize(cum_memory_length * fused_depth);
        float* fused_proj = _projections[1](memory, cum_memory_length);
        std::vector<float*> splits = split_in_depth(fused_proj, cum_memory_length, fused_depth,
                                                    2, _splits.data());
        keys_proj = splits[0];
        values_proj = splits[1];
        _cached_memory_keys = keys_proj;
        _cached_memory_values = values_proj;
      }
    }

    unsigned int dk = _depth / _num_heads;
    array_mul(1.0 / sqrt(dk), queries_proj, cum_queries_length * _depth);

    _padded_queries.maybe_resize(batch_size * queries_time * _depth);
    _padded_keys.maybe_resize(batch_size * memory_time * _depth);
    _padded_values.maybe_resize(batch_size * memory_time * _depth);
    pad_sequences(queries_proj, queries_length, batch_size, queries_time, _depth, _padded_queries.data());
    pad_sequences(keys_proj, memory_length, batch_size, memory_time, _depth, _padded_keys.data());
    pad_sequences(values_proj, memory_length, batch_size, memory_time, _depth, _padded_values.data());

    _split_queries.maybe_resize(batch_size * queries_time * _depth);
    _split_keys.maybe_resize(batch_size * memory_time * _depth);
    _split_values.maybe_resize(batch_size * memory_time * _depth);
    swap_middle_dims(_padded_queries.data(), batch_size, queries_time, _num_heads, dk, _split_queries.data());
    swap_middle_dims(_padded_keys.data(), batch_size, memory_time, _num_heads, dk, _split_keys.data());
    swap_middle_dims(_padded_values.data(), batch_size, memory_time, _num_heads, dk, _split_values.data());

    float* context = _attention(_split_queries.data(),
                                _split_keys.data(),
                                _split_values.data(),
                                batch_size * _num_heads,
                                queries_time,
                                memory_time,
                                dk);

    float* combined = _padded_queries.data();
    float* combined_pruned = queries_proj;
    swap_middle_dims(context, batch_size, _num_heads, queries_time, dk, combined);
    unpad_sequences(combined, queries_length, batch_size, queries_time, _depth, combined_pruned);

    output = _projections.back()(combined_pruned, cum_queries_length, output);

    array_add(queries, output, cum_queries_length * _projections.back().output_depth());
    return output;
  }

private:
  unsigned int _num_heads;
  unsigned int _depth;
  LayerNorm _layer_norm;
  DotProductAttention _attention;
  ReusableVector<float> _splits;
  std::vector<Dense> _projections;
  ReusableVector<float> _keys_accu;
  ReusableVector<float> _values_accu;
  const float* _cached_memory_keys = nullptr;
  const float* _cached_memory_values = nullptr;
  ReusableVector<float> _padded_queries;
  ReusableVector<float> _padded_keys;
  ReusableVector<float> _padded_values;
  ReusableVector<float> _split_queries;
  ReusableVector<float> _split_keys;
  ReusableVector<float> _split_values;
};

class TransformerEncoderLayer : public Node
{
public:
  TransformerEncoderLayer(const Model& model,
                          const std::string& scope)
    : _multi_head_attention(model, scope + "/multi_head", 8)
    , _ff(model, scope + "/ffn") {
  }

  float* operator()(const float* input,
                    const float* memory,
                    const unsigned int* lengths,
                    const unsigned int* memory_lengths,
                    unsigned int batch_size,
                    float* output = nullptr) {
    unsigned int total_batch_size = 0;
    for (unsigned int i = 0; i < batch_size; ++i)
      total_batch_size += lengths[i];
    const float* context = _multi_head_attention(input, memory, lengths, memory_lengths, batch_size);
    return _ff(context, total_batch_size, output);
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

  float* operator()(unsigned int step,
                    const float* input,
                    const float* memory,
                    const unsigned int* lengths,
                    const unsigned int* history_lengths,
                    const unsigned int* memory_lengths,
                    unsigned int batch_size,
                    float* output = nullptr) {
    const float* encoded = _masked_multi_head_attention(
      input, nullptr, lengths, history_lengths, batch_size, step);
    const float* context = _multi_head_attention(
      encoded, memory, lengths, memory_lengths, batch_size, step);
    return _ff(context, batch_size, output);
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
    , _position_encoder(_scaled_embeddings.output_depth())
    , _output_norm(model, scope + "/LayerNorm") {
    for (unsigned int l = 0;; ++l) {
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

  float* operator()(const unsigned int* ids,
                    const unsigned int* lengths,
                    unsigned int batch_size,
                    unsigned int flattened_batch_size,
                    float* output = nullptr) {
    const float* embeddings = _scaled_embeddings(ids, flattened_batch_size);
    const float* input = _position_encoder(embeddings, lengths, batch_size, flattened_batch_size);

    const float* x = input;
    for (auto& layer : _layers) {
      x = layer(x, nullptr, lengths, nullptr, batch_size);
    }
    return _output_norm(x, flattened_batch_size, output);
  }
};

class TransformerDecoder : public TransformerStack<TransformerDecoderLayer>
{
public:
  TransformerDecoder(const Model& model, const std::string& scope)
    : TransformerStack(model, scope)
    , _proj(model, scope + "/dense") {
  }

  float* operator()(unsigned int step,
                    const unsigned int* ids,
                    unsigned int batch_size,
                    const float* memory,
                    const unsigned int* memory_length,
                    float* output = nullptr) {
    const float* embeddings = _scaled_embeddings(ids, batch_size);
    const float* input = _position_encoder(embeddings, step, batch_size);
    std::vector<unsigned int> query_lengths(batch_size, 1);
    std::vector<unsigned int> history_lengths(batch_size, step + 1);

    const float* x = input;
    for (auto& layer : _layers)
      x = layer(step, x, memory, query_lengths.data(), history_lengths.data(), memory_length, batch_size);
    x = _output_norm(x, batch_size);
    return _proj(x, batch_size, output);
  }

private:
  Dense _proj;
};

void translate(const std::vector<std::vector<std::string> >& input_tokens,
               const Vocabulary& vocabulary,
               TransformerEncoder& encoder,
               TransformerDecoder& decoder) {
  std::vector<unsigned int> ids;
  std::vector<unsigned int> lengths;
  unsigned int max_length = 0;
  unsigned int cum_length = 0;
  for (const auto& sentence : input_tokens) {
    unsigned int length = 0;
    for (const auto& token : sentence) {
      ids.push_back(vocabulary.to_id(token));
      ++length;
    }
    if (length > max_length)
      max_length = length;
    cum_length += length;
    lengths.push_back(length);
  }

  // for (auto id : ids)
  //   std::cout << id << std::endl;

  unsigned int batch_size = lengths.size();
  const float* encoded = encoder(ids.data(), lengths.data(), batch_size, cum_length);

  std::vector<unsigned int> sample_from(batch_size, vocabulary.to_id("<s>"));
  std::vector<std::vector<unsigned int> > sampled_ids(batch_size);
  std::vector<bool> finished(batch_size, false);
  std::vector<float> probs(batch_size * vocabulary.size());
  bool all_finished = false;
  unsigned max_steps = 200;

  for (unsigned int step = 0; !all_finished; ++step) {
    float* logits = decoder(step, sample_from.data(), batch_size, encoded, lengths.data());
    softmax(logits, batch_size, vocabulary.size(), probs.data());
    all_finished = true;
    for (unsigned int i = 0; i < batch_size; ++i) {
      unsigned int best = cblas_isamax(vocabulary.size(), probs.data() + (i*vocabulary.size()), 1);
      sample_from[i] = best;
      if (best == 2 || (step + 1) == max_steps)
        finished[i] = true;
      else {
        all_finished = false;
        sampled_ids[i].push_back(best);
      }
    }
  }

  for (unsigned int i = 0; i < batch_size; ++i) {
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
  unsigned int max_batch_size = 1;
  unsigned int max_iter = 10000;
  unsigned int iter = 0;

  while (std::getline(text_file, line)) {
    std::cout << "  INPUT: " << line << std::endl;
    input_tokens.emplace_back();
    std::string token;
    for (unsigned int i = 0; i < line.length(); ++i) {
      if (line[i] == ' ') {
        if (!token.empty()) {
          input_tokens.back().push_back(token);
          token.clear();
        }
      } else {
        token += line[i];
      }
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
