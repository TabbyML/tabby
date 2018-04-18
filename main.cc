#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <cstring>
#include <unordered_map>

#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>
#include <mkl.h>
// #include <mkldnn.hpp>

#include "model.h"
#include "vocabulary.h"
#include "routines.h"

using Tensor2D = Eigen::Tensor<float, 2, Eigen::RowMajor>;
using Tensor3D = Eigen::Tensor<float, 3, Eigen::RowMajor>;
using Tensor4D = Eigen::Tensor<float, 4, Eigen::RowMajor>;

void maybe_resize(std::vector<float>& v, size_t size) {
  if (size > v.size()) {
    v.resize(size);
  }
}

class Node
{
public:
  virtual ~Node() {
    free(_output);
  }
protected:
  float* output_buffer(size_t size) {
    if (size > _alloc_size) {
      _output = realloc(_output, size * sizeof (float));
      _alloc_size = size;
    }
    return reinterpret_cast<float*>(_output);
  }
private:
  void* _output = nullptr;
  size_t _alloc_size = 0;
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
    maybe_resize(_tmp, _depth);
    for (unsigned int i = 0; i < batch_size; ++i) {
      const float* x = input + (i * _depth);
      float* y = output + (i * _depth);
      float mean = array_mean(x, _depth);
      array_copy(x, y, _depth);
      array_sub(mean, y, _depth); // y is now centered
      array_pow(y, _tmp.data(), 2, _depth);
      float variance = array_mean(_tmp.data(), _depth);
      array_mul(1.0 / sqrt(variance), y, _depth); // y is now centered and normalized.
      array_mul(_gamma, y, _depth);
      array_add(_beta, y, _depth);
    }
    return output;
  }

private:
  const float* _beta;
  const float* _gamma;
  unsigned int _depth;
  std::vector<float> _tmp;
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
    maybe_resize(_dot, batch_size * queries_time * keys_time);
    batch_mat_mul(queries, keys,
                  CblasNoTrans, CblasTrans,
                  batch_size, queries_time, keys_time, depth,
                  _dot.data());
    maybe_resize(_attn, batch_size * queries_time * keys_time);
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
  std::vector<float> _dot;
  std::vector<float> _attn;
};

class MultiHeadAttention
{
public:
  MultiHeadAttention(const Model& model,
                     const std::string& scope,
                     unsigned int num_heads,
                     bool with_cache)
    : _num_heads(num_heads)
    , _layer_norm(model, scope + "/LayerNorm")
    , _with_cache(with_cache) {
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

    //std::cout << normed_queries[0] << std::endl;

    if (memory == nullptr) {
      float* fused_proj = _projections[0](normed_queries, cum_queries_length);
      unsigned int fused_depth = _projections[0].output_depth();
      maybe_resize(_splits, cum_queries_length * fused_depth);
      std::vector<float*> splits = split_in_depth(fused_proj, cum_queries_length, fused_depth,
                                                  3, _splits.data());
      queries_proj = splits[0];
      keys_proj = splits[1];
      values_proj = splits[2];
      if (_with_cache) {
        //std::cout << "cache previous self proj" << std::endl;
        _step += 1;
        maybe_resize(_keys_accu, _step * batch_size * _depth);
        maybe_resize(_values_accu, _step * batch_size * _depth);
        array_copy(keys_proj, &_keys_accu.back() + 1 - batch_size * _depth, batch_size * _depth);
        array_copy(values_proj, &_values_accu.back() + 1 - batch_size * _depth, batch_size * _depth);
        keys_proj = _keys_accu.data();
        values_proj = _values_accu.data();
      }
    } else {
      queries_proj = _projections[0](normed_queries, cum_queries_length);
      if (_with_cache && _cached_memory_keys != nullptr) {
        //std::cout << "cache encoder memory proj" << std::endl;
        keys_proj = _cached_memory_keys;
        values_proj = _cached_memory_values;
      } else {
        unsigned int fused_depth = _projections[1].output_depth();
        maybe_resize(_splits, cum_memory_length * fused_depth);
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

    std::vector<float> padded_queries(batch_size * queries_time * _depth);
    std::vector<float> padded_keys(batch_size * memory_time * _depth);
    std::vector<float> padded_values(batch_size * memory_time * _depth);
    pad_sequences(queries_proj, queries_length, batch_size, queries_time, _depth, padded_queries.data());
    pad_sequences(keys_proj, memory_length, batch_size, memory_time, _depth, padded_keys.data());
    pad_sequences(values_proj, memory_length, batch_size, memory_time, _depth, padded_values.data());

    // b x T x D
    Eigen::TensorMap<Tensor4D> t_queries_map(padded_queries.data(), batch_size, queries_time, _num_heads, dk);
    Eigen::TensorMap<Tensor4D> t_keys_map(padded_keys.data(), batch_size, memory_time, _num_heads, dk);
    Eigen::TensorMap<Tensor4D> t_values_map(padded_values.data(), batch_size, memory_time, _num_heads, dk);

    Eigen::array<int, 4> shuffling;
    shuffling[0] = 0;
    shuffling[1] = 2;
    shuffling[2] = 1;
    shuffling[3] = 3;

    Tensor4D t_queries = t_queries_map.shuffle(shuffling);
    Tensor4D t_keys = t_keys_map.shuffle(shuffling);
    Tensor4D t_values = t_values_map.shuffle(shuffling);

    //std::cout << queries_time << " " << memory_time << std::endl;

    float* context = _attention(t_queries.data(),
                                t_keys.data(),
                                t_values.data(),
                                batch_size * _num_heads,
                                queries_time,
                                memory_time,
                                dk);

    //std::cout << "context: " << context[0] << std::endl;

    Eigen::TensorMap<Tensor4D> t_context(context, batch_size, _num_heads, queries_time, dk);
    Tensor4D t_combined = t_context.shuffle(shuffling);

    //std::cout << "combined: " << t_combined.data()[0] << std::endl << std::endl;

    std::vector<float> combined_pruned(cum_queries_length * _depth);
    unpad_sequences(t_combined.data(), queries_length, batch_size, queries_time, _depth, combined_pruned.data());

    output = _projections.back()(combined_pruned.data(), cum_queries_length, output);

    array_add(queries, output, cum_queries_length * _projections.back().output_depth());
    return output;
  }

private:
  unsigned int _num_heads;
  unsigned int _depth;
  LayerNorm _layer_norm;
  DotProductAttention _attention;
  std::vector<float> _splits;
  std::vector<Dense> _projections;
  bool _with_cache;
  std::vector<float> _keys_accu;
  std::vector<float> _values_accu;
  unsigned int _step = 0;
  const float* _cached_memory_keys = nullptr;
  const float* _cached_memory_values = nullptr;
};

class TransformerEncoderLayer : public Node
{
public:
  TransformerEncoderLayer(const Model& model,
                          const std::string& scope)
    : _multi_head_attention(model, scope + "/multi_head", 8, false)
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
    : _masked_multi_head_attention(model, scope + "/masked_multi_head", 8, true)
    , _multi_head_attention(model, scope + "/multi_head", 8, true)
    , _ff(model, scope + "/ffn") {
  }

  float* operator()(const float* input,
                    const float* memory,
                    const unsigned int* lengths,
                    const unsigned int* history_lengths,
                    const unsigned int* memory_lengths,
                    unsigned int batch_size,
                    float* output = nullptr) {
    const float* encoded = _masked_multi_head_attention(
      input, nullptr, lengths, history_lengths, batch_size);
    const float* context = _multi_head_attention(
      encoded, memory, lengths, memory_lengths, batch_size);
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
  TransformerStack(const Model& model, const std::string& scope, bool dynamic)
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
    : TransformerStack(model, scope, false) {
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
      //std::cout << x[0] << std::endl;
    }
    return _output_norm(x, flattened_batch_size, output);
  }
};

class TransformerDecoder : public TransformerStack<TransformerDecoderLayer>
{
public:
  TransformerDecoder(const Model& model, const std::string& scope)
    : TransformerStack(model, scope, true)
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
      x = layer(x, memory, query_lengths.data(), history_lengths.data(), memory_length, batch_size);
    x = _output_norm(x, batch_size);
    return _proj(x, batch_size, output);
  }

private:
  Dense _proj;
};

int main() {
  Model model("/home/klein/dev/ctransformer/model.bin");
  Vocabulary vocabulary("/home/klein/data/wmt-ende/wmtende.vocab");

  std::vector<std::vector<std::string> > input = {
    {"▁Gut", "ach", ":", "▁Increase", "d", "▁safety", "▁for", "▁pedestrian", "s"}//,
    //   {"▁They", "▁are", "▁not", "▁even", "▁100", "▁metres", "▁apart"}
  };

  // Ref Trans:
  // ▁Gut ach : ▁Mehr ▁Sicherheit ▁für ▁Fußgänger
  // ▁Sie ▁liegen ▁nicht ▁einmal ▁100 ▁Meter ▁voneinander ▁entfernt

  std::vector<unsigned int> ids;
  std::vector<unsigned int> lengths;
  unsigned int max_length = 0;
  unsigned int cum_length = 0;
  for (const auto& sentence : input) {
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

  // for (const auto& id : ids)
  //   std::cout << id << std::endl;

  unsigned int batch_size = lengths.size();

  TransformerEncoder encoder(model, "transformer/encoder");
  const float* encoded = encoder(ids.data(), lengths.data(), batch_size, cum_length);

  TransformerDecoder decoder(model, "transformer/decoder");

  std::vector<unsigned int> sample_from = { 1, 1 };
  std::vector<std::vector<unsigned int> > sampled_ids(batch_size);
  std::vector<bool> finished(batch_size, false);
  std::vector<float> probs(batch_size * vocabulary.size());
  bool all_finished = false;

  for (unsigned int step = 0; !all_finished; ++step) {
    float* logits = decoder(step, sample_from.data(), batch_size, encoded, lengths.data());
    softmax(logits, batch_size, vocabulary.size(), probs.data());
    all_finished = true;
    for (unsigned int i = 0; i < batch_size; ++i) {
      unsigned int best = cblas_isamax(vocabulary.size(), probs.data() + (i*vocabulary.size()), 1);
      std::cout << i << ": " << vocabulary.to_token(best) << std::endl;
      sample_from[i] = best;
      if (best == 2)
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

  return 0;
}
