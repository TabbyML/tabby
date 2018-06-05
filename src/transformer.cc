#include "opennmt/transformer.h"

#include <fstream>

namespace opennmt {

  template <typename T>
  T consume(std::ifstream& in) {
    T val;
    in.read(reinterpret_cast<char*>(&val), sizeof (T));
    return val;
  }

  template <typename T>
  T* consume(std::ifstream& in, size_t n) {
    T* data = new T[n];
    in.read(reinterpret_cast<char*>(data), n * sizeof (T));
    return data;
  }

  Model::Model(const std::string& path) {
    std::ifstream model(path, std::ios_base::in | std::ios_base::binary);
    if (!model.is_open())
      throw std::runtime_error("failed to load the model " + path);

    auto num_variables = consume<uint32_t>(model);

    for (uint32_t i = 0; i < num_variables; ++i) {
      auto name_length = consume<uint16_t>(model);
      auto name = consume<char>(model, name_length);
      auto rank = consume<uint8_t>(model);
      auto dimensions = consume<uint32_t>(model, rank);
      auto data_width = consume<uint8_t>(model);
      auto data_size = consume<uint32_t>(model);
      auto data = consume<char>(model, data_size * data_width);

      std::vector<size_t> shape(rank);
      for (unsigned int k = 0; k < rank; k++) {
        shape[k] = static_cast<size_t>(dimensions[k]);
      }

      StorageView* view = nullptr;

      if (data_width == 4) {
        view = new StorageView(reinterpret_cast<float*>(data), shape);
      } else if (data_width == 2) {
        view = new StorageView(reinterpret_cast<int16_t*>(data), shape);
      }

      _variable_index.emplace(std::piecewise_construct,
                              std::forward_as_tuple(name),
                              std::forward_as_tuple(*view));

      delete view;
      delete [] name;
      delete [] dimensions;
      delete [] data;
    }
  }

  const StorageView& Model::get_variable(const std::string& scope) const {
    auto it = _variable_index.lower_bound(scope);
    if (it->first.find(scope) == std::string::npos)
      throw std::out_of_range("no variable found in scope '" + scope + "'");
    return it->second;
  }


  ScaledEmbeddings::ScaledEmbeddings(const Model& model, const std::string& scope)
    : _embeddings(model.get_variable(scope + "/w_embs"))
    , _gathered(_embeddings.dtype()) {
  }

  StorageView& ScaledEmbeddings::operator()(const StorageView& ids) {
    if (_embeddings.dtype() == DataType::DT_FLOAT) {
      _gather_op(_embeddings, ids, _output);
    } else {
      _gather_op(_embeddings, ids, _gathered);
      ops::Unquantize(1000)(_gathered, _output);
    }
    const size_t embedding_size = _embeddings.dim(-1);
    primitives::mul(static_cast<float>(sqrt(embedding_size)),
                    _output.data<float>(),
                    _output.size());
    return _output;
  }


  StorageView& PositionEncoder::operator()(const StorageView& input,
                                           const StorageView& lengths) {
    assert(input.rank() == 3);
    size_t depth = input.dim(-1);
    if (_cached_encodings.empty())
      precompute_position_encoding(_max_cached_time, depth);
    _output = input;
    for (size_t i = 0; i < lengths.dim(0); ++i) {
      const auto length = lengths.at<int32_t>(i);
      primitives::add(_cached_encodings.data<float>(),
                      _output.index<float>({i}),
                      length * depth);
    }
    return _output;
  }

  StorageView& PositionEncoder::operator()(const StorageView& input, size_t index) {
    size_t depth = input.dim(-1);
    if (_cached_encodings.empty())
      precompute_position_encoding(_max_cached_time, depth);
    _output = input;
    for (size_t i = 0; i < input.dim(0); ++i) {
      primitives::add(_cached_encodings.index<float>({index}),
                      _output.index<float>({i}),
                      depth);
    }
    return _output;
  }

  void PositionEncoder::precompute_position_encoding(size_t max_time, size_t depth) {
    float log_timescale_increment = log(10000) / (depth / 2 - 1);
    StorageView timescales({depth / 2}, -log_timescale_increment);
    for (size_t i = 0; i < timescales.size(); ++i)
      timescales.data<float>()[i] = exp(timescales.data<float>()[i] * i);

    StorageView scaled_time({max_time, depth / 2});
    for (size_t i = 0; i < scaled_time.dim(0); ++i) {
      for (size_t j = 0; j < scaled_time.dim(1); ++j) {
        *scaled_time.index<float>({i, j}) = (i + 1) * timescales.data<float>()[j];
      }
    }

    StorageView sin_encoding;
    StorageView cos_encoding;

    ops::Sin()(scaled_time, sin_encoding);
    ops::Cos()(scaled_time, cos_encoding);
    ops::Concat(-1)({&sin_encoding, &cos_encoding}, _cached_encodings);
  }


  Dense::Dense(const Model& model, const std::string& scope)
    : _gemm_op(1, 1, true, false, true)
    , _weight(model.get_variable(scope + "/kernel"))
    , _bias(model.get_variable(scope + "/bias")) {
  }

  StorageView& Dense::operator()(const StorageView& input) {
    _gemm_op(input, _weight, _bias, _output);
    return _output;
  }

  LayerNorm::LayerNorm(const Model& model, const std::string& scope)
    : _beta(model.get_variable(scope + "/beta"))
    , _gamma(model.get_variable(scope + "/gamma")) {
  }

  StorageView& LayerNorm::operator()(const StorageView& input) {
    _norm_op(_beta, _gamma, input, _output);
    return _output;
  }


  TransformerFeedForward::TransformerFeedForward(const Model& model,
                                                 const std::string& scope)
    : _layer_norm(model, scope + "/LayerNorm")
    , _ff1(model, scope + "/conv1d")
    , _ff2(model, scope + "/conv1d_1") {
  }

  StorageView& TransformerFeedForward::operator()(const StorageView& input) {
    const StorageView& normed = _layer_norm(input);
    StorageView& inner = _ff1(normed);
    ops::ReLU()(inner);
    StorageView& outer = _ff2(inner);
    primitives::add(input.data<float>(), outer.data<float>(), input.size());
    return outer;
  }


  StorageView& DotProductAttention::operator()(const StorageView& queries,
                                               const StorageView& keys,
                                               const StorageView& values,
                                               const StorageView& values_lengths) {
    assert(queries.rank() == 4);
    assert(keys.rank() == 4);
    assert(values.rank() == 4);

    size_t batch_size = queries.dim(0);
    size_t num_heads = queries.dim(1);
    size_t queries_time = queries.dim(2);
    size_t memory_time = keys.dim(2);

    ops::MatMul(false, true)(queries, keys, _dot);

    if (batch_size > 1) {
      for (size_t b = 0; b < batch_size; ++b) {
        const size_t length = values_lengths.data<int32_t>()[b];
        if (length == memory_time)
          continue;
        for (size_t h = 0; h < num_heads; ++h) {
          for (size_t i = 0; i < queries_time; ++i) {
            auto* x = _dot.index<float>({b, h, i});
            primitives::fill(x + length, std::numeric_limits<float>::lowest(), memory_time - length);
          }
        }
      }
    }

    ops::SoftMax()(_dot, _attn);
    ops::MatMul()(_attn, values, _dot);
    return _dot;
  }


  MultiHeadAttention::MultiHeadAttention(const Model& model,
                                         const std::string& scope,
                                         size_t num_heads)
    : _layer_norm(model, scope + "/LayerNorm")
    , _num_heads(num_heads)
    , _transpose_op({0, 2, 1, 3}) {
  }

  void MultiHeadAttention::split_heads(const StorageView& x, StorageView& y) {
    StorageView z(const_cast<float*>(x.data<float>()),
                  {x.dim(0), x.dim(1), _num_heads, x.dim(2) / _num_heads});
    _transpose_op(z, y);
  }

  void MultiHeadAttention::combine_heads(const StorageView& x, StorageView& y) {
    _transpose_op(x, y);
    y.reshape({y.dim(0), y.dim(1), y.dim(-1) * _num_heads});
  }

  StorageView& MultiHeadAttention::compute_attention(const StorageView& queries,
                                                     const StorageView& keys,
                                                     const StorageView& values,
                                                     const StorageView& queries_lengths,
                                                     const StorageView& values_lengths) {
    split_heads(queries, _split_queries);
    split_heads(keys, _split_keys);
    split_heads(values, _split_values);

    const size_t dk = queries.dim(-1) / _num_heads;
    primitives::mul(static_cast<float>(1.0 / sqrt(dk)),
                    _split_queries.data<float>(),
                    _split_queries.size());

    const StorageView& context = _attention(_split_queries,
                                            _split_keys,
                                            _split_values,
                                            values_lengths);

    combine_heads(context, _combined);
    return _combined;
  }


  TransformerSelfAttention::TransformerSelfAttention(const Model& model,
                                                     const std::string& scope,
                                                     size_t num_heads)
    : MultiHeadAttention(model, scope, num_heads)
    , _linear_in(model, scope + "/conv1d")
    , _linear_out(model, scope + "/conv1d_1") {
  }

  StorageView& TransformerSelfAttention::operator()(const StorageView& queries,
                                                    const StorageView& queries_lengths,
                                                    StorageView* cached_keys,
                                                    StorageView* cached_values,
                                                    ssize_t step) {
    const StorageView& normed_queries = _layer_norm(queries);
    const StorageView& fused_proj = _linear_in(normed_queries);
    ops::Split(-1)(fused_proj, _queries_proj, _keys_proj, _values_proj);

    StorageView values_lengths(queries_lengths);
    StorageView keys_proj;
    StorageView values_proj;

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

    const StorageView& attention_output = compute_attention(_queries_proj,
                                                            keys_proj,
                                                            values_proj,
                                                            queries_lengths,
                                                            values_lengths);

    StorageView& output = _linear_out(attention_output);
    primitives::add(queries.data<float>(), output.data<float>(), queries.size());
    return output;
  }

  void TransformerSelfAttention::cache_proj(ssize_t step, StorageView& proj, StorageView& cache) {
    if (step == 0) {
      cache = proj;
    } else {
      static StorageView tmp;
      tmp = cache;
      ops::Concat(1)({&tmp, &proj}, cache);
    }
  }


  TransformerAttention::TransformerAttention(const Model& model,
                                             const std::string& scope,
                                             size_t num_heads)
    : MultiHeadAttention(model, scope, num_heads)
    , _linear_query(model, scope + "/conv1d")
    , _linear_memory(model, scope + "/conv1d_1")
    , _linear_out(model, scope + "/conv1d_2") {
  }

  StorageView& TransformerAttention::operator()(const StorageView& queries,
                                                const StorageView& queries_lengths,
                                                const StorageView& memory,
                                                const StorageView& memory_lengths,
                                                StorageView* cached_keys,
                                                StorageView* cached_values,
                                                ssize_t step) {
    const StorageView& normed_queries = _layer_norm(queries);
    const StorageView& queries_proj = _linear_query(normed_queries);

    if (step > 0 && cached_keys != nullptr && !cached_keys->empty()) {
      _keys_proj.shallow_copy(*cached_keys);
      _values_proj.shallow_copy(*cached_values);
    } else {
      const StorageView& memory_proj = _linear_memory(memory);
      ops::Split(-1)(memory_proj, _keys_proj, _values_proj);
      if (cached_keys != nullptr) {
        *cached_keys = _keys_proj;
        *cached_values = _values_proj;
      }
    }

    const StorageView& attention_output = compute_attention(queries_proj,
                                                            _keys_proj,
                                                            _values_proj,
                                                            queries_lengths,
                                                            memory_lengths);

    StorageView& output = _linear_out(attention_output);
    primitives::add(queries.data<float>(), output.data<float>(), queries.size());
    return output;
  }


  TransformerEncoderLayer::TransformerEncoderLayer(const Model& model,
                                                   const std::string& scope)
    : _self_attention(model, scope + "/multi_head", 8)
    , _ff(model, scope + "/ffn") {
  }

  StorageView& TransformerEncoderLayer::operator()(const StorageView& input,
                                                   const StorageView& lengths) {
    const auto& context = _self_attention(input, lengths);
    return _ff(context);
  }


  TransformerDecoderLayer::TransformerDecoderLayer(const Model& model,
                                                   const std::string& scope)
    : _self_attention(model, scope + "/masked_multi_head", 8)
    , _encoder_attention(model, scope + "/multi_head", 8)
    , _ff(model, scope + "/ffn") {
  }

  StorageView& TransformerDecoderLayer::operator()(size_t step,
                                                   const StorageView& input,
                                                   const StorageView& input_lengths,
                                                   const StorageView& memory,
                                                   const StorageView& memory_lengths,
                                                   StorageView& cached_self_attn_keys,
                                                   StorageView& cached_self_attn_values,
                                                   StorageView& cached_attn_keys,
                                                   StorageView& cached_attn_values) {
    const auto& encoded = _self_attention(
      input, input_lengths, &cached_self_attn_keys, &cached_self_attn_values, step);
    const auto& context = _encoder_attention(
      encoded, input_lengths, memory, memory_lengths, &cached_attn_keys, &cached_attn_values, step);
    return _ff(context);
  }


  TransformerEncoder::TransformerEncoder(const Model& model, const std::string& scope)
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

  StorageView& TransformerEncoder::encode(const StorageView& ids, const StorageView& lengths) {
    const auto& embeddings = _scaled_embeddings(ids);
    const auto& input = _position_encoder(embeddings, lengths);
    const auto* x = &input;
    for (auto& layer : _layers) {
      x = &layer(*x, lengths);
    }
    return _output_norm(*x);
  }


  TransformerDecoderState::TransformerDecoderState(size_t num_layers)
    : DecoderState() {
    for (size_t i = 0; i < num_layers; ++i) {
      add("self_keys_" + std::to_string(i));
      add("self_values_" + std::to_string(i));
      add("memory_keys_" + std::to_string(i));
      add("memory_values_" + std::to_string(i));
    }
  }

  void TransformerDecoderState::reset(const StorageView& memory,
                                      const StorageView& memory_lengths) {
    DecoderState::reset(memory, memory_lengths);
    for (auto& pair : _states) {
      if (pair.first != "memory" and pair.first != "memory_lengths")
        pair.second.clear();
    }
  }


  TransformerDecoder::TransformerDecoder(const Model& model, const std::string& scope)
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

  StorageView& TransformerDecoder::logits(size_t step, const StorageView& ids) {
    size_t batch_size = ids.dim(0);
    const auto& embeddings = _scaled_embeddings(ids);
    const auto& input = _position_encoder(embeddings, step);
    StorageView query_lengths({batch_size}, static_cast<int32_t>(1));

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

}
