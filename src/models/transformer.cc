#include "ctranslate2/models/transformer.h"

#include <fstream>

namespace ctranslate2 {
  namespace models {

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

    TransformerModel::TransformerModel(const std::string& path, Device device)
      : Model(path, device) {
      std::string model_path = path + "/model.bin";
      std::ifstream model(model_path, std::ios_base::in | std::ios_base::binary);
      if (!model.is_open())
        throw std::runtime_error("failed to load the model " + model_path);

      _version = consume<uint32_t>(model);
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

        _variable_index.emplace(std::piecewise_construct,
                                std::forward_as_tuple(name),
                                std::forward_as_tuple(load_data(shape, data_width, data)));

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

    std::unique_ptr<Encoder> TransformerModel::make_encoder() const {
      return std::unique_ptr<Encoder>(new TransformerEncoder(*this, "transformer/encoder"));
    }

    std::unique_ptr<Decoder> TransformerModel::make_decoder() const {
      return std::unique_ptr<Decoder>(new TransformerDecoder(*this, "transformer/decoder"));
    }

    size_t TransformerModel::version() const {
      return _version;
    }


    ScaledEmbeddings::ScaledEmbeddings(const TransformerModel& model, const std::string& scope)
      : _embeddings(model.get_variable(scope + "/w_embs"))
      , _scale(static_cast<float>(sqrt(_embeddings.dim(-1)))) {
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
      ops::Mul()(output, _scale, output);
    }


    PositionEncoder::PositionEncoder(const TransformerModel& model, const std::string& scope) {
      try {
        encoding = &model.get_variable(scope + "/pe");
      } catch (std::out_of_range&) {
        encoding = nullptr;
      }
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
      if (encoding)
        return *encoding;

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
      : _weight(model.get_variable(scope + "/kernel"))
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
        static const ops::Quantize quantize_op(1000);
        static const ops::Unquantize unquantize_op(1000 * 1000);
        static thread_local StorageView quantized_input(_weight.dtype());
        static thread_local StorageView quantized_output(DataType::DT_INT32);
        quantize_op(input, quantized_input);
        gemm_op(quantized_input, *weight, *bias, quantized_output);
        unquantize_op(quantized_output, output);
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
      : _layer_norm(model, scope + "/LayerNorm")
      , _ff1(model, scope + "/conv1d")
      , _ff2(model, scope + "/conv1d_1") {
    }

    void TransformerFeedForward::operator()(const StorageView& input, StorageView& output) {
      static thread_local StorageView inner(input.device());
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
        static thread_local StorageView output_host;
        static thread_local StorageView lengths_host(DataType::DT_INT32);
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

      static thread_local StorageView attn(values.device());
      ops::SoftMax()(output, attn);
      ops::MatMul()(attn, values, output);
    }


    MultiHeadAttention::MultiHeadAttention(const TransformerModel& model,
                                           const std::string& scope,
                                           size_t num_heads)
      : _num_heads(num_heads)
      , _layer_norm(model, scope + "/LayerNorm")
      , _transpose_op({0, 2, 1, 3}) {
      for (size_t i = 0;; ++i) {
        try {
          _linear.emplace_back(model, scope + "/conv1d" + (i > 0 ? "_" + std::to_string(i) : ""));
        } catch (std::exception&) {
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

      static thread_local StorageView normed_queries(device);
      _layer_norm(queries, normed_queries);

      static thread_local StorageView fused_proj(device);
      _linear[0](normed_queries, fused_proj);

      static thread_local StorageView queries_proj(device);
      static thread_local StorageView keys_proj(device);
      static thread_local StorageView values_proj(device);
      static thread_local StorageView split_queries(device);
      static thread_local StorageView split_keys(device);
      static thread_local StorageView split_values(device);

      StorageView final_queries(device);
      StorageView final_keys(device);
      StorageView final_values(device);

      if (memory) {
        split_heads(fused_proj, split_queries);
        final_queries.shallow_copy(split_queries);
        if (cached_keys != nullptr && !cached_keys->empty()) {
          final_keys.shallow_copy(*cached_keys);
          final_values.shallow_copy(*cached_values);
        } else {
          _linear[1](*memory, fused_proj);
          ops::Split(-1)(fused_proj, keys_proj, values_proj);
          split_heads(keys_proj, split_keys);
          split_heads(values_proj, split_values);
          final_keys.shallow_copy(split_keys);
          final_values.shallow_copy(split_values);
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
        final_queries.shallow_copy(split_queries);
        if (cached_keys != nullptr) {
          cache_proj(split_keys, *cached_keys);
          cache_proj(split_values, *cached_values);
          final_keys.shallow_copy(*cached_keys);
          final_values.shallow_copy(*cached_values);
        } else {
          final_keys.shallow_copy(split_keys);
          final_values.shallow_copy(split_values);
        }
      }

      const size_t dk = queries.dim(-1) / _num_heads;
      const StorageView scale(static_cast<float>(1.0 / sqrt(dk)));
      ops::Mul()(final_queries, scale, final_queries);

      StorageView& context = queries_proj;  // Reuse storage.
      _attention(final_queries,
                 final_keys,
                 final_values,
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
        static thread_local StorageView tmp(proj.device());
        tmp = std::move(cache);
        ops::Concat(2)({&tmp, &proj}, cache);
      }
    }


    TransformerEncoderLayer::TransformerEncoderLayer(const TransformerModel& model,
                                                     const std::string& scope)
      : _self_attention(model, scope + "/multi_head", 8)
      , _ff(model, scope + "/ffn") {
    }

    void TransformerEncoderLayer::operator()(const StorageView& input,
                                             const StorageView& lengths,
                                             StorageView& output) {
      static thread_local StorageView context(input.device());
      _self_attention(input, nullptr, &lengths, context);
      _ff(context, output);
    }


    TransformerDecoderLayer::TransformerDecoderLayer(const TransformerModel& model,
                                                     const std::string& scope)
      : _self_attention(model, scope + "/masked_multi_head", 8)
      , _encoder_attention(model, scope + "/multi_head", 8)
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
      static thread_local StorageView context(input.device());
      _self_attention(input, nullptr, nullptr, output,
                      &cached_self_attn_keys, &cached_self_attn_values);
      _encoder_attention(output, &memory, &memory_lengths, context,
                         &cached_attn_keys, &cached_attn_values);
      return _ff(context, output);
    }


    TransformerEncoder::TransformerEncoder(const TransformerModel& model, const std::string& scope)
      : _scaled_embeddings(model, scope)
      , _position_encoder(model, scope)
      , _output_norm(model, scope + "/LayerNorm") {
      for (size_t l = 0;; ++l) {
        try {
          _layers.emplace_back(model, scope + "/layer_" + std::to_string(l));
        } catch (std::exception&) {
          break;
        }
      }
    }

    void TransformerEncoder::operator()(const StorageView& ids,
                                        const StorageView& lengths,
                                        StorageView& output) {
      static thread_local StorageView layer_in(output.device());
      static thread_local StorageView layer_out(output.device());
      _scaled_embeddings(ids, layer_in);
      _position_encoder(layer_in);

      for (auto& layer : _layers) {
        layer(layer_in, lengths, layer_out);
        std::swap(layer_in, layer_out);
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
      : _scaled_embeddings(model, scope)
      , _position_encoder(model, scope)
      , _output_norm(model, scope + "/LayerNorm")
      , _proj(model, scope + "/dense") {
      for (size_t l = 0;; ++l) {
        try {
          _layers.emplace_back(model, scope + "/layer_" + std::to_string(l));
        } catch (std::exception&) {
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
      static thread_local StorageView layer_in(output.device());
      static thread_local StorageView layer_out(output.device());

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
        std::swap(layer_in, layer_out);
      }
      _output_norm(layer_in, layer_out);

      _proj(layer_out, output, &candidates);
    }

  }
}
