#include "ctranslate2/models/transformer.h"

namespace ctranslate2 {
  namespace models {

    static bool replace(std::string& str, const std::string& from, const std::string& to) {
      size_t start_pos = str.find(from);
      if (start_pos == std::string::npos)
        return false;
      str.replace(start_pos, from.length(), to);
      return true;
    }

    static std::string map_v1_variable_name(std::string name) {
      // V1 variable names were simply the names defined by OpenNMT-tf.
      replace(name, "transformer/", "");
      replace(name, ":0", "");
      replace(name, "w_embs", "embeddings/weight");
      replace(name, "kernel", "weight");
      replace(name, "LayerNorm", "layer_norm");
      replace(name, "dense", "projection");
      replace(name, "conv1d_", "linear_");
      replace(name, "conv1d", "linear_0");
      if (name.find("encoder") != std::string::npos) {
        replace(name, "multi_head", "self_attention");
      } else {
        replace(name, "masked_multi_head", "self_attention");
        replace(name, "multi_head", "attention");
      }
      return name;
    }

    TransformerModel::TransformerModel(const std::string& path,
                                       size_t spec_revision,
                                       Device device,
                                       size_t num_heads)
      : Model(path, spec_revision, device)
      , _num_heads(num_heads) {
    }

    size_t TransformerModel::num_heads() const {
      return _num_heads;
    }

    size_t TransformerModel::current_spec_revision() const {
      return 2;
    }

    void TransformerModel::register_variable(const std::string& name, StorageView& variable) {
      std::string var_name = name;
      if (_spec_revision == 1)
        var_name = map_v1_variable_name(name);
      Model::register_variable(var_name, variable);
    }

    std::unique_ptr<Encoder> TransformerModel::make_encoder() const {
      return std::unique_ptr<Encoder>(new TransformerEncoder(*this, "encoder"));
    }

    std::unique_ptr<Decoder> TransformerModel::make_decoder() const {
      return std::unique_ptr<Decoder>(new TransformerDecoder(*this, "decoder"));
    }


    TransformerBaseModel::TransformerBaseModel(const std::string& path,
                                               size_t spec_revision,
                                               Device device)
      : TransformerModel(path, spec_revision, device, 8) {
    }

    TransformerBigModel::TransformerBigModel(const std::string& path,
                                             size_t spec_revision,
                                             Device device)
      : TransformerModel(path, spec_revision, device, 16) {
    }


    ScaledEmbeddings::ScaledEmbeddings(const TransformerModel& model, const std::string& scope)
      : _embeddings(model.get_variable(scope + "/weight"))
      , _qscale(model.get_variable_if_exists(scope + "/weight_scale"))
      , _scale(static_cast<float>(sqrt(_embeddings.dim(-1)))) {
    }

    void ScaledEmbeddings::operator()(const StorageView& ids,
                                      StorageView& output) {
      if (_embeddings.dtype() == DataType::DT_INT16) {
        StorageView gathered(_embeddings.dtype());
        _gather_op(_embeddings, ids, gathered);
        ops::Unquantize()(gathered, *_qscale, output);
      } else {
        _gather_op(_embeddings, ids, output);
      }
      ops::Mul()(output, _scale, output);
    }


    PositionEncoder::PositionEncoder(const TransformerModel& model, const std::string& scope)
      : _encoding(model.get_variable_if_exists(scope + "/encodings")) {
    }

    void PositionEncoder::operator()(StorageView& input, size_t index) {
      const size_t max_time = input.dim(1);
      const size_t depth = input.dim(-1);
      const StorageView& encodings = get_position_encoding(max_time, depth, input.device());
      DEVICE_DISPATCH(input.device(),
                      primitives<D>::add_batch_broadcast(encodings.data<float>() + index * depth,
                                                         input.data<float>(),
                                                         max_time * depth,
                                                         input.size()));
    }

    const StorageView& PositionEncoder::get_position_encoding(size_t max_time,
                                                              size_t depth,
                                                              Device device) const {
      if (_encoding)
        return *_encoding;

      static const size_t default_max_time = 500;
      static thread_local StorageView position_encoding(device);

      if (position_encoding.empty() || max_time > position_encoding.dim(0)) {
        size_t reserved_time = (position_encoding.empty()
                                ? std::max(default_max_time, max_time)
                                : max_time);
        float log_timescale_increment = log(10000) / (depth / 2 - 1);
        StorageView timescales({depth / 2}, -log_timescale_increment);
        for (size_t i = 0; i < timescales.size(); ++i)
          timescales.data<float>()[i] = exp(timescales.data<float>()[i] * i);

        StorageView scaled_time({reserved_time, depth / 2});
        for (size_t i = 0; i < scaled_time.dim(0); ++i) {
          for (size_t j = 0; j < scaled_time.dim(1); ++j) {
            *scaled_time.index<float>({i, j}) = (i + 1) * timescales.data<float>()[j];
          }
        }

        StorageView sin_encoding;
        StorageView cos_encoding;

        ops::Sin()(scaled_time, sin_encoding);
        ops::Cos()(scaled_time, cos_encoding);

        StorageView cache;
        ops::Concat(-1)({&sin_encoding, &cos_encoding}, cache);
        position_encoding = cache.to(device);
      }

      return position_encoding;
    }

    Dense::Dense(const TransformerModel& model, const std::string& scope)
      : _weight(model.get_variable(scope + "/weight"))
      , _qscale(model.get_variable_if_exists(scope + "/weight_scale"))
      , _bias(model.get_variable(scope + "/bias"))
      , _partial_weight(_weight.device(), _weight.dtype())
      , _partial_bias(_bias.device(), _bias.dtype()) {
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

      static const ops::Gemm gemm_op(1, 0, false, false, true);
      if (_weight.dtype() == DataType::DT_FLOAT) {
        gemm_op(input, *weight, *bias, output);
      } else {
        StorageView quantized_input(_weight.dtype());
        StorageView quantized_output(DataType::DT_INT32);
        StorageView squared_scale(_qscale->as_scalar<float>() * _qscale->as_scalar<float>());
        ops::Quantize()(input, *_qscale, quantized_input);
        gemm_op(quantized_input, *weight, *bias, quantized_output);
        ops::Unquantize()(quantized_output, squared_scale, output);
      }

      DEVICE_DISPATCH(output.device(),
                      primitives<D>::add_batch_broadcast(bias->data<float>(),
                                                         output.data<float>(),
                                                         bias->size(),
                                                         output.size()));
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
      : _layer_norm(model, scope + "/layer_norm")
      , _ff1(model, scope + "/linear_0")
      , _ff2(model, scope + "/linear_1") {
    }

    void TransformerFeedForward::operator()(const StorageView& input, StorageView& output) {
      StorageView inner(input.device());
      _layer_norm(input, output);
      _ff1(output, inner);
      ops::ReLU()(inner, inner);
      _ff2(inner, output);
      ops::Add()(input, output, output);
    }


    void DotProductAttention::operator()(const StorageView& queries,
                                         const StorageView& keys,
                                         const StorageView& values,
                                         const StorageView* values_lengths,
                                         StorageView& output) {
      assert(queries.rank() == 4);
      assert(keys.rank() == 4);
      assert(values.rank() == 4);

      size_t batch_size = queries.dim(0);
      size_t num_heads = queries.dim(1);
      size_t queries_time = queries.dim(2);
      size_t memory_time = keys.dim(2);

      ops::MatMul(false, true)(queries, keys, output);

      if (values_lengths && batch_size > 1) {
        StorageView output_host;
        StorageView lengths_host(DataType::DT_INT32);
        output_host.copy_from(output);
        lengths_host.copy_from(*values_lengths);
        for (size_t b = 0; b < batch_size; ++b) {
          const size_t length = lengths_host.data<int32_t>()[b];
          if (length == memory_time)
            continue;
          for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < queries_time; ++i) {
              auto* x = output_host.index<float>({b, h, i});
              DEVICE_DISPATCH(output_host.device(),
                              primitives<D>::fill(x + length,
                                                  std::numeric_limits<float>::lowest(),
                                                  memory_time - length));
            }
          }
        }
        output.copy_from(output_host);
      }

      StorageView attn(values.device());
      ops::SoftMax()(output, attn);
      ops::MatMul()(attn, values, output);
    }


    MultiHeadAttention::MultiHeadAttention(const TransformerModel& model, const std::string& scope)
      : _num_heads(model.num_heads())
      , _layer_norm(model, scope + "/layer_norm")
      , _transpose_op({0, 2, 1, 3}) {
      for (size_t i = 0;; ++i) {
        try {
          _linear.emplace_back(model, scope + "/linear_" + std::to_string(i));
        } catch (std::exception&) {
          if (i == 0)
            throw;
          else
            break;
        }
      }
    }

    void MultiHeadAttention::operator()(const StorageView& queries,
                                        const StorageView* memory,
                                        const StorageView* memory_lengths,
                                        StorageView& output,
                                        StorageView* cached_keys,
                                        StorageView* cached_values) {
      Device device = queries.device();
      StorageView fused_proj(device);
      StorageView queries_proj(device);
      StorageView keys_proj(device);
      StorageView values_proj(device);
      StorageView split_queries(device);
      StorageView split_keys(device);
      StorageView split_values(device);

      _layer_norm(queries, queries_proj);
      _linear[0](queries_proj, fused_proj);

      if (memory) {
        split_heads(fused_proj, split_queries);
        if (cached_keys != nullptr && !cached_keys->empty()) {
          split_keys.shallow_copy(*cached_keys);
          split_values.shallow_copy(*cached_values);
        } else {
          _linear[1](*memory, fused_proj);
          ops::Split(-1)(fused_proj, keys_proj, values_proj);
          split_heads(keys_proj, split_keys);
          split_heads(values_proj, split_values);
          if (cached_keys != nullptr) {
            *cached_keys = split_keys;
            *cached_values = split_values;
          }
        }
      } else {
        ops::Split(-1)(fused_proj, queries_proj, keys_proj, values_proj);
        split_heads(queries_proj, split_queries);
        split_heads(keys_proj, split_keys);
        split_heads(values_proj, split_values);
        if (cached_keys != nullptr) {
          cache_proj(split_keys, *cached_keys);
          cache_proj(split_values, *cached_values);
          split_keys.shallow_copy(*cached_keys);
          split_values.shallow_copy(*cached_values);
        }
      }

      const size_t dk = queries.dim(-1) / _num_heads;
      const StorageView scale(static_cast<float>(1.0 / sqrt(dk)));
      ops::Mul()(split_queries, scale, split_queries);

      StorageView& context = queries_proj;  // Reuse storage.
      _attention(split_queries,
                 split_keys,
                 split_values,
                 memory_lengths,
                 context);

      StorageView& combined = values_proj;  // Reuse storage.
      combine_heads(context, combined);

      _linear.back()(combined, output);
      ops::Add()(queries, output, output);
    }

    void MultiHeadAttention::split_heads(const StorageView& x, StorageView& y) {
      StorageView z({x.dim(0), x.dim(1), _num_heads, x.dim(2) / _num_heads},
                    const_cast<float*>(x.data<float>()), x.device());
      _transpose_op(z, y);
    }

    void MultiHeadAttention::combine_heads(const StorageView& x, StorageView& y) {
      _transpose_op(x, y);
      y.reshape({y.dim(0), y.dim(1), y.dim(-1) * _num_heads});
    }

    void MultiHeadAttention::cache_proj(StorageView& proj, StorageView& cache) {
      if (cache.empty()) {
        cache = proj;
      } else {
        StorageView tmp(proj.device());
        tmp = std::move(cache);
        ops::Concat(2)({&tmp, &proj}, cache);
      }
    }


    TransformerEncoderLayer::TransformerEncoderLayer(const TransformerModel& model,
                                                     const std::string& scope)
      : _self_attention(model, scope + "/self_attention")
      , _ff(model, scope + "/ffn") {
    }

    void TransformerEncoderLayer::operator()(const StorageView& input,
                                             const StorageView& lengths,
                                             StorageView& output) {
      StorageView context(input.device());
      _self_attention(input, nullptr, &lengths, context);
      _ff(context, output);
    }


    TransformerDecoderLayer::TransformerDecoderLayer(const TransformerModel& model,
                                                     const std::string& scope)
      : _self_attention(model, scope + "/self_attention")
      , _encoder_attention(model, scope + "/attention")
      , _ff(model, scope + "/ffn") {
    }

    void TransformerDecoderLayer::operator()(const StorageView& input,
                                             const StorageView& memory,
                                             const StorageView& memory_lengths,
                                             StorageView& cached_self_attn_keys,
                                             StorageView& cached_self_attn_values,
                                             StorageView& cached_attn_keys,
                                             StorageView& cached_attn_values,
                                             StorageView& output) {
      StorageView context(input.device());
      _self_attention(input, nullptr, nullptr, output,
                      &cached_self_attn_keys, &cached_self_attn_values);
      _encoder_attention(output, &memory, &memory_lengths, context,
                         &cached_attn_keys, &cached_attn_values);
      return _ff(context, output);
    }


    TransformerEncoder::TransformerEncoder(const TransformerModel& model, const std::string& scope)
      : _scaled_embeddings(model, scope + "/embeddings")
      , _position_encoder(model, scope + "/position_encodings")
      , _output_norm(model, scope + "/layer_norm") {
      for (size_t l = 0;; ++l) {
        try {
          _layers.emplace_back(model, scope + "/layer_" + std::to_string(l));
        } catch (std::exception&) {
          if (l == 0)
            throw;
          else
            break;
        }
      }
    }

    void TransformerEncoder::operator()(const StorageView& ids,
                                        const StorageView& lengths,
                                        StorageView& output) {
      StorageView layer_in(output.device());
      StorageView layer_out(output.device());
      _scaled_embeddings(ids, layer_in);
      _position_encoder(layer_in);

      for (auto& layer : _layers) {
        layer(layer_in, lengths, layer_out);
        swap(layer_in, layer_out);
      }
      _output_norm(layer_in, output);
    }


    TransformerDecoderState::TransformerDecoderState(size_t num_layers, Device device)
      : _num_layers(num_layers)
      , _device(device) {
    }

    void TransformerDecoderState::reset() {
      static const StorageView empty_cache(_device);
      for (size_t i = 0; i < _num_layers; ++i) {
        reset_state("self_keys_" + std::to_string(i), empty_cache);
        reset_state("self_values_" + std::to_string(i), empty_cache);
        reset_state("memory_keys_" + std::to_string(i), empty_cache);
        reset_state("memory_values_" + std::to_string(i), empty_cache);
      }
    }


    TransformerDecoder::TransformerDecoder(const TransformerModel& model, const std::string& scope)
      : _scaled_embeddings(model, scope + "/embeddings")
      , _position_encoder(model, scope + "/position_encodings")
      , _output_norm(model, scope + "/layer_norm")
      , _proj(model, scope + "/projection") {
      for (size_t l = 0;; ++l) {
        try {
          _layers.emplace_back(model, scope + "/layer_" + std::to_string(l));
        } catch (std::exception&) {
          if (l == 0)
            throw;
          else
            break;
        }
      }
      _state.reset(new TransformerDecoderState(_layers.size(), model.device()));
    }

    void TransformerDecoder::operator()(size_t step,
                                        const StorageView& ids,
                                        const StorageView& candidates,
                                        const StorageView& memory,
                                        const StorageView& memory_lengths,
                                        StorageView& output) {
      StorageView layer_in(output.device());
      StorageView layer_out(output.device());

      _scaled_embeddings(ids, layer_in);
      _position_encoder(layer_in, step);

      for (size_t l = 0; l < _layers.size(); ++l) {
        _layers[l](layer_in,
                   memory,
                   memory_lengths,
                   _state->get("self_keys_" + std::to_string(l)),
                   _state->get("self_values_" + std::to_string(l)),
                   _state->get("memory_keys_" + std::to_string(l)),
                   _state->get("memory_values_" + std::to_string(l)),
                   layer_out);
        swap(layer_in, layer_out);
      }
      _output_norm(layer_in, layer_out);

      _proj(layer_out, output, &candidates);
    }

  }
}
