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

  TransformerModel::TransformerModel(const std::string& path)
    : _source_vocabulary(path + "/source_vocabulary.txt")
    , _target_vocabulary(path + "/target_vocabulary.txt") {
    std::string model_path = path + "/model.bin";
    std::ifstream model(model_path, std::ios_base::in | std::ios_base::binary);
    if (!model.is_open())
      throw std::runtime_error("failed to load the model " + model_path);

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

      // View the read buffer.
      StorageView* view = nullptr;
      if (data_width == 4) {
        view = new StorageView(shape, reinterpret_cast<float*>(data));
      } else if (data_width == 2) {
        view = new StorageView(shape, reinterpret_cast<int16_t*>(data));
      }

      // We use the copy constructor so that the storage owns aligned data.
      _variable_index.emplace(std::piecewise_construct,
                              std::forward_as_tuple(name),
                              std::forward_as_tuple(*view));

      delete view;
      delete [] name;
      delete [] dimensions;
      delete [] data;
    }
  }

  const StorageView& TransformerModel::get_variable(const std::string& scope) const {
    auto it = _variable_index.lower_bound(scope);
    if (it->first.find(scope) == std::string::npos)
      throw std::out_of_range("no variable found in scope '" + scope + "'");
    return it->second;
  }

  const Vocabulary& TransformerModel::get_source_vocabulary() const {
    return _source_vocabulary;
  }

  const Vocabulary& TransformerModel::get_target_vocabulary() const {
    return _source_vocabulary;
  }

  std::unique_ptr<Encoder> TransformerModel::make_encoder() const {
    return std::unique_ptr<Encoder>(new TransformerEncoder(*this, "transformer/encoder"));
  }

  std::unique_ptr<Decoder> TransformerModel::make_decoder() const {
    return std::unique_ptr<Decoder>(new TransformerDecoder(*this, "transformer/decoder"));
  }


  ScaledEmbeddings::ScaledEmbeddings(const TransformerModel& model, const std::string& scope)
    : _embeddings(model.get_variable(scope + "/w_embs")) {
  }

  void ScaledEmbeddings::operator()(const StorageView& ids,
                                    StorageView& output) {
    if (_embeddings.dtype() == DataType::DT_INT16) {
      static const ops::Unquantize unquantize_op(1000);
      static thread_local StorageView gathered(_embeddings.dtype());
      _gather_op(_embeddings, ids, gathered);
      unquantize_op(gathered, output);
    } else {
      _gather_op(_embeddings, ids, output);
    }
    const size_t embedding_size = _embeddings.dim(-1);
    primitives::mul(static_cast<float>(sqrt(embedding_size)),
                    output.data<float>(),
                    output.size());
  }


  void PositionEncoder::operator()(const StorageView& input,
                                   const StorageView& lengths,
                                   StorageView& output) {
    assert(input.rank() == 3);
    size_t depth = input.dim(-1);
    if (_cached_encodings.empty())
      precompute_position_encoding(_max_cached_time, depth);
    output = input;
    for (size_t i = 0; i < lengths.dim(0); ++i) {
      const auto length = lengths.at<int32_t>(i);
      primitives::add(_cached_encodings.data<float>(),
                      output.index<float>({i}),
                      length * depth);
    }
  }

  void PositionEncoder::operator()(const StorageView& input,
                                   size_t index,
                                   StorageView& output) {
    size_t depth = input.dim(-1);
    if (_cached_encodings.empty())
      precompute_position_encoding(_max_cached_time, depth);
    output = input;
    for (size_t i = 0; i < input.dim(0); ++i) {
      primitives::add(_cached_encodings.index<float>({index}),
                      output.index<float>({i}),
                      depth);
    }
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

  Dense::Dense(const TransformerModel& model, const std::string& scope)
    : _weight(model.get_variable(scope + "/kernel"))
    , _bias(model.get_variable(scope + "/bias"))
    , _partial_weight(_weight.dtype())
    , _partial_bias(_bias.dtype()) {
  }

  void Dense::operator()(const StorageView& input,
                         StorageView& output,
                         const StorageView* index) {
    const StorageView* weight = &_weight;
    const StorageView* bias = &_bias;
    if (index && !index->empty()) {
      ops::Gather()(_weight, *index, _partial_weight);
      ops::Gather()(_bias, *index, _partial_bias);
      weight = &_partial_weight;
      bias = &_partial_bias;
    }

    if (_weight.dtype() == DataType::DT_FLOAT) {
      static const ops::Gemm gemm_op(1, 1, true, false, true);
      gemm_op(input, *weight, *bias, output);
    } else {
      static const ops::Gemm gemm_op(1, 0, false, false, true);
      static const ops::Quantize quantize_op(1000);
      static const ops::Unquantize unquantize_op(1000 * 1000);
      static thread_local StorageView quantized_input(_weight.dtype());
      static thread_local StorageView quantized_output(DataType::DT_INT32);
      quantize_op(input, quantized_input);
      gemm_op(quantized_input, *weight, *bias, quantized_output);
      unquantize_op(quantized_output, output);
      size_t output_depth = bias->size();
      for (size_t i = 0; i < output.size() / output_depth; ++i)
        primitives::add(bias->data<float>(), output.data<float>() + i * output_depth, output_depth);
    }
  }


  LayerNorm::LayerNorm(const TransformerModel& model, const std::string& scope)
    : _beta(model.get_variable(scope + "/beta"))
    , _gamma(model.get_variable(scope + "/gamma")) {
  }

  void LayerNorm::operator()(const StorageView& input, StorageView& output) {
    _norm_op(_beta, _gamma, input, output);
  }


  TransformerFeedForward::TransformerFeedForward(const TransformerModel& model,
                                                 const std::string& scope)
    : _layer_norm(model, scope + "/LayerNorm")
    , _ff1(model, scope + "/conv1d")
    , _ff2(model, scope + "/conv1d_1") {
  }

  void TransformerFeedForward::operator()(const StorageView& input, StorageView& output) {
    static thread_local StorageView inner;
    _layer_norm(input, output);
    _ff1(output, inner);
    ops::ReLU()(inner);
    _ff2(inner, output);
    primitives::add(input.data<float>(), output.data<float>(), input.size());
  }


  void DotProductAttention::operator()(const StorageView& queries,
                                       const StorageView& keys,
                                       const StorageView& values,
                                       const StorageView& values_lengths,
                                       StorageView& output) {
    assert(queries.rank() == 4);
    assert(keys.rank() == 4);
    assert(values.rank() == 4);

    size_t batch_size = queries.dim(0);
    size_t num_heads = queries.dim(1);
    size_t queries_time = queries.dim(2);
    size_t memory_time = keys.dim(2);

    ops::MatMul(false, true)(queries, keys, output);

    if (batch_size > 1) {
      for (size_t b = 0; b < batch_size; ++b) {
        const size_t length = values_lengths.data<int32_t>()[b];
        if (length == memory_time)
          continue;
        for (size_t h = 0; h < num_heads; ++h) {
          for (size_t i = 0; i < queries_time; ++i) {
            auto* x = output.index<float>({b, h, i});
            primitives::fill(x + length, std::numeric_limits<float>::lowest(), memory_time - length);
          }
        }
      }
    }

    static thread_local StorageView attn;
    ops::SoftMax()(output, attn);
    ops::MatMul()(attn, values, output);
  }


  MultiHeadAttention::MultiHeadAttention(const TransformerModel& model,
                                         const std::string& scope,
                                         size_t num_heads)
    : _layer_norm(model, scope + "/LayerNorm")
    , _num_heads(num_heads)
    , _transpose_op({0, 2, 1, 3}) {
  }

  void MultiHeadAttention::split_heads(const StorageView& x, StorageView& y) {
    StorageView z({x.dim(0), x.dim(1), _num_heads, x.dim(2) / _num_heads},
                  const_cast<float*>(x.data<float>()));
    _transpose_op(z, y);
  }

  void MultiHeadAttention::combine_heads(const StorageView& x, StorageView& y) {
    _transpose_op(x, y);
    y.reshape({y.dim(0), y.dim(1), y.dim(-1) * _num_heads});
  }

  void MultiHeadAttention::compute_attention(const StorageView& queries,
                                             const StorageView& keys,
                                             const StorageView& values,
                                             const StorageView& values_lengths,
                                             StorageView& output) {
    static thread_local StorageView split_queries;
    static thread_local StorageView split_keys;
    static thread_local StorageView split_values;

    split_heads(queries, split_queries);
    split_heads(keys, split_keys);
    split_heads(values, split_values);

    const size_t dk = queries.dim(-1) / _num_heads;
    primitives::mul(static_cast<float>(1.0 / sqrt(dk)),
                    split_queries.data<float>(),
                    split_queries.size());

    static thread_local StorageView context;
    _attention(split_queries,
               split_keys,
               split_values,
               values_lengths,
               context);
    combine_heads(context, output);
  }


  TransformerSelfAttention::TransformerSelfAttention(const TransformerModel& model,
                                                     const std::string& scope,
                                                     size_t num_heads)
    : MultiHeadAttention(model, scope, num_heads)
    , _linear_in(model, scope + "/conv1d")
    , _linear_out(model, scope + "/conv1d_1") {
  }

  void TransformerSelfAttention::operator()(const StorageView& queries,
                                            const StorageView& queries_lengths,
                                            StorageView& output,
                                            StorageView* cached_keys,
                                            StorageView* cached_values,
                                            ssize_t step) {
    static thread_local StorageView normed_queries;
    static thread_local StorageView fused_proj;
    static thread_local StorageView queries_proj;
    static thread_local StorageView keys_proj;
    static thread_local StorageView values_proj;

    _layer_norm(queries, normed_queries);
    _linear_in(normed_queries, fused_proj);
    ops::Split(-1)(fused_proj, queries_proj, keys_proj, values_proj);

    StorageView values_lengths(queries_lengths);
    StorageView true_keys_proj;
    StorageView true_values_proj;

    if (step >= 0 && cached_keys != nullptr) {
      cache_proj(step, keys_proj, *cached_keys);
      cache_proj(step, values_proj, *cached_values);
      true_keys_proj.shallow_copy(*cached_keys);
      true_values_proj.shallow_copy(*cached_values);
      values_lengths.fill(static_cast<int32_t>(step + 1));
    } else {
      true_keys_proj.shallow_copy(keys_proj);
      true_values_proj.shallow_copy(values_proj);
    }

    static thread_local StorageView context;
    compute_attention(queries_proj,
                      true_keys_proj,
                      true_values_proj,
                      values_lengths,
                      context);

    _linear_out(context, output);
    primitives::add(queries.data<float>(), output.data<float>(), queries.size());
  }

  void TransformerSelfAttention::cache_proj(ssize_t step, StorageView& proj, StorageView& cache) {
    if (step == 0) {
      cache = proj;
    } else {
      static thread_local StorageView tmp;
      tmp = cache;
      ops::Concat(1)({&tmp, &proj}, cache);
    }
  }


  TransformerAttention::TransformerAttention(const TransformerModel& model,
                                             const std::string& scope,
                                             size_t num_heads)
    : MultiHeadAttention(model, scope, num_heads)
    , _linear_query(model, scope + "/conv1d")
    , _linear_memory(model, scope + "/conv1d_1")
    , _linear_out(model, scope + "/conv1d_2") {
  }

  void TransformerAttention::operator()(const StorageView& queries,
                                        const StorageView& memory,
                                        const StorageView& memory_lengths,
                                        StorageView& output,
                                        StorageView* cached_keys,
                                        StorageView* cached_values,
                                        ssize_t step) {
    static thread_local StorageView normed_queries;
    static thread_local StorageView memory_proj;
    static thread_local StorageView queries_proj;
    static thread_local StorageView keys_proj;
    static thread_local StorageView values_proj;

    _layer_norm(queries, normed_queries);
    _linear_query(normed_queries, queries_proj);

    if (step > 0 && cached_keys != nullptr && !cached_keys->empty()) {
      keys_proj.shallow_copy(*cached_keys);
      values_proj.shallow_copy(*cached_values);
    } else {
      _linear_memory(memory, memory_proj);
      ops::Split(-1)(memory_proj, keys_proj, values_proj);
      if (cached_keys != nullptr) {
        *cached_keys = keys_proj;
        *cached_values = values_proj;
      }
    }

    static thread_local StorageView context;
    compute_attention(queries_proj,
                      keys_proj,
                      values_proj,
                      memory_lengths,
                      context);

    _linear_out(context, output);
    primitives::add(queries.data<float>(), output.data<float>(), queries.size());
  }


  TransformerEncoderLayer::TransformerEncoderLayer(const TransformerModel& model,
                                                   const std::string& scope)
    : _self_attention(model, scope + "/multi_head", 8)
    , _ff(model, scope + "/ffn") {
  }

  void TransformerEncoderLayer::operator()(const StorageView& input,
                                           const StorageView& lengths,
                                           StorageView& output) {
    static thread_local StorageView context;
    _self_attention(input, lengths, context);
    _ff(context, output);
  }


  TransformerDecoderLayer::TransformerDecoderLayer(const TransformerModel& model,
                                                   const std::string& scope)
    : _self_attention(model, scope + "/masked_multi_head", 8)
    , _encoder_attention(model, scope + "/multi_head", 8)
    , _ff(model, scope + "/ffn") {
  }

  void TransformerDecoderLayer::operator()(size_t step,
                                           const StorageView& input,
                                           const StorageView& input_lengths,
                                           const StorageView& memory,
                                           const StorageView& memory_lengths,
                                           StorageView& cached_self_attn_keys,
                                           StorageView& cached_self_attn_values,
                                           StorageView& cached_attn_keys,
                                           StorageView& cached_attn_values,
                                           StorageView& output) {
    static thread_local StorageView context;
    _self_attention(input, input_lengths, output,
                    &cached_self_attn_keys, &cached_self_attn_values, step);
    _encoder_attention(output, memory, memory_lengths, context,
                       &cached_attn_keys, &cached_attn_values, step);
    return _ff(context, output);
  }


  TransformerEncoder::TransformerEncoder(const TransformerModel& model, const std::string& scope)
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

  void TransformerEncoder::encode(const StorageView& ids,
                                  const StorageView& lengths,
                                  StorageView& output) {
    static thread_local StorageView embeddings;
    static thread_local StorageView layer_in;
    static thread_local StorageView layer_out;
    _scaled_embeddings(ids, embeddings);
    _position_encoder(embeddings, lengths, layer_in);
    for (auto& layer : _layers) {
      layer(layer_in, lengths, layer_out);
      std::swap(layer_in, layer_out);
    }
    _output_norm(layer_in, output);
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


  TransformerDecoder::TransformerDecoder(const TransformerModel& model, const std::string& scope)
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

  void TransformerDecoder::logits(size_t step,
                                  const StorageView& ids,
                                  const StorageView& candidates,
                                  StorageView& output) {
    static thread_local StorageView embeddings;
    static thread_local StorageView layer_in;
    static thread_local StorageView layer_out;

    size_t batch_size = ids.dim(0);
    _scaled_embeddings(ids, embeddings);
    _position_encoder(embeddings, step, layer_in);
    StorageView query_lengths({batch_size}, static_cast<int32_t>(1));

    for (size_t l = 0; l < _layers.size(); ++l) {
      _layers[l](step,
                 layer_in,
                 query_lengths,
                 _state->get("memory"),
                 _state->get("memory_lengths"),
                 _state->get("self_keys_" + std::to_string(l)),
                 _state->get("self_values_" + std::to_string(l)),
                 _state->get("memory_keys_" + std::to_string(l)),
                 _state->get("memory_values_" + std::to_string(l)),
                 layer_out);
      std::swap(layer_in, layer_out);
    }
    _output_norm(layer_in, layer_out);
    _proj(layer_out, output, &candidates);
  }

}
