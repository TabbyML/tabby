#include "ctranslate2/models/transformer.h"

#include <cmath>

#include "../device_dispatch.h"

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
                                       size_t num_heads)
      : SequenceToSequenceModel(path, spec_revision)
      , _num_heads(num_heads) {
    }

    size_t TransformerModel::num_heads() const {
      return _num_heads;
    }

    bool TransformerModel::with_relative_position() const {
      return _with_relative_position;
    }

    size_t TransformerModel::current_spec_revision() const {
      return 3;
    }

    void TransformerModel::register_variable(const std::string& name, StorageView& variable) {
      std::string var_name = name;
      if (_spec_revision == 1)
        var_name = map_v1_variable_name(name);
      SequenceToSequenceModel::register_variable(var_name, variable);
    }

    void TransformerModel::finalize() {
      SequenceToSequenceModel::finalize();
      if (_spec_revision >= 3)
        _num_heads = get_variable("num_heads").as_scalar<int8_t>();
      _with_relative_position = get_flag_with_default("with_relative_position", false);
    }

    std::unique_ptr<layers::Encoder> TransformerModel::make_encoder() const {
      return std::unique_ptr<layers::Encoder>(new TransformerEncoder(*this, "encoder"));
    }

    std::unique_ptr<layers::Decoder> TransformerModel::make_decoder() const {
      return std::unique_ptr<layers::Decoder>(new TransformerDecoder(*this, "decoder"));
    }


    PositionEncoder::PositionEncoder()
      : _model_encoding(nullptr) {
    }

    PositionEncoder::PositionEncoder(const TransformerModel& model, const std::string& scope)
      : _model_encoding(model.get_variable_if_exists(scope + "/encodings")) {
    }

    void PositionEncoder::operator()(StorageView& input, dim_t index) {
      const dim_t max_time = input.dim(1);
      const dim_t depth = input.dim(-1);
      const StorageView& encodings = get_position_encoding(max_time, depth, input.device());
      DEVICE_DISPATCH(input.device(),
                      primitives<D>::add_batch_broadcast(encodings.data<float>() + index * depth,
                                                         input.data<float>(),
                                                         max_time * depth,
                                                         input.size()));
    }

    const StorageView& PositionEncoder::get_position_encoding(dim_t max_time,
                                                              dim_t depth,
                                                              Device device) {
      if (_model_encoding)
        return *_model_encoding;

      static const dim_t default_max_time = 500;
      if (!_generated_encoding)
        _generated_encoding.reset(new StorageView(device));

      if (_generated_encoding->empty() || max_time > _generated_encoding->dim(0)) {
        dim_t reserved_time = (_generated_encoding->empty()
                               ? std::max(default_max_time, max_time)
                               : max_time);
        float log_timescale_increment = log(10000) / (depth / 2 - 1);
        StorageView timescales({depth / 2}, -log_timescale_increment);
        for (dim_t i = 0; i < timescales.size(); ++i)
          timescales.data<float>()[i] = exp(timescales.data<float>()[i] * i);

        StorageView scaled_time({reserved_time, depth / 2});
        for (dim_t i = 0; i < scaled_time.dim(0); ++i) {
          for (dim_t j = 0; j < scaled_time.dim(1); ++j) {
            *scaled_time.index<float>({i, j}) = (i + 1) * timescales.data<float>()[j];
          }
        }

        StorageView sin_encoding;
        StorageView cos_encoding;

        ops::Sin()(scaled_time, sin_encoding);
        ops::Cos()(scaled_time, cos_encoding);

        StorageView cache;
        ops::Concat(-1)({&sin_encoding, &cos_encoding}, cache);
        *_generated_encoding = cache.to(device);
      }

      return *_generated_encoding;
    }


    TransformerFeedForward::TransformerFeedForward(const TransformerModel& model,
                                                   const std::string& scope)
      : _layer_norm(model, scope + "/layer_norm")
      , _ff1(model, scope + "/linear_0")
      , _ff2(model, scope + "/linear_1") {
    }

    void TransformerFeedForward::operator()(const StorageView& input, StorageView& output) const {
      StorageView inner(input.device());
      _layer_norm(input, output);
      _ff1(output, inner);
      ops::ReLU()(inner, inner);
      _ff2(inner, output);
      ops::Add()(input, output, output);
    }


    TransformerEncoderLayer::TransformerEncoderLayer(const TransformerModel& model,
                                                     const std::string& scope)
      : _self_attention(model,
                        scope + "/self_attention",
                        model.num_heads(),
                        /*self_attention=*/true)
      , _ff(model, scope + "/ffn") {
    }

    void TransformerEncoderLayer::operator()(const StorageView& input,
                                             const StorageView& lengths,
                                             StorageView& output) const {
      PROFILE("TransformerEncoderLayer");
      StorageView context(input.device());
      _self_attention(input, nullptr, &lengths, context);
      _ff(context, output);
    }


    TransformerDecoderLayer::TransformerDecoderLayer(const TransformerModel& model,
                                                     const std::string& scope,
                                                     const bool with_encoder_attention)
      : _self_attention(model,
                        scope + "/self_attention",
                        model.num_heads(),
                        /*self_attention=*/true)
      , _encoder_attention(with_encoder_attention
                           ? new layers::MultiHeadAttention(model,
                                                            scope + "/attention",
                                                            model.num_heads(),
                                                            /*self_attention=*/false)
                           : nullptr)
      , _ff(model, scope + "/ffn") {
    }

    void TransformerDecoderLayer::operator()(const StorageView& input,
                                             const StorageView* memory,
                                             const StorageView* memory_lengths,
                                             StorageView& cached_self_attn_keys,
                                             StorageView& cached_self_attn_values,
                                             StorageView* cached_attn_keys,
                                             StorageView* cached_attn_values,
                                             StorageView& output,
                                             StorageView* attention) const {
      PROFILE("TransformerDecoderLayer");
      StorageView context(input.device());
      if (_encoder_attention) {
        _self_attention(input, nullptr, nullptr, output,
                        &cached_self_attn_keys, &cached_self_attn_values);
        (*_encoder_attention)(output, memory, memory_lengths, context,
                              cached_attn_keys, cached_attn_values, attention);
      } else {
        _self_attention(input, nullptr, nullptr, context,
                        &cached_self_attn_keys, &cached_self_attn_values);
      }
      _ff(context, output);
    }


    TransformerEncoder::TransformerEncoder(const TransformerModel& model, const std::string& scope)
      : _embeddings(model, scope + "/embeddings")
      , _position_encoder(model.with_relative_position()
                          ? nullptr
                          : new PositionEncoder(model, scope + "/position_encodings"))
      , _output_norm(model, scope + "/layer_norm") {
      for (size_t l = 0;; ++l) {
        try {
          _layers.emplace_back(new TransformerEncoderLayer(model,
                                                           scope + "/layer_" + std::to_string(l)));
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
      PROFILE("TransformerEncoder");
      StorageView input(output.device());
      _embeddings(ids, input);
      if (_position_encoder)
        (*_position_encoder)(input);

      for (size_t l = 0; l < _layers.size(); ++l) {
        (*_layers[l])(input, lengths, output);
        if (l + 1 < _layers.size())
          swap(input, output);
      }
      _output_norm(output, output);
    }


    TransformerDecoder::TransformerDecoder(const TransformerModel& model,
                                           const std::string& scope,
                                           const bool with_encoder_attention)
      : Decoder(model.device())
      , _with_encoder_attention(with_encoder_attention)
      , _embeddings(model, scope + "/embeddings")
      , _position_encoder(model.with_relative_position()
                          ? nullptr
                          : new PositionEncoder(model, scope + "/position_encodings"))
      , _output_norm(model, scope + "/layer_norm")
      , _proj(model, scope + "/projection") {
      for (size_t l = 0;; ++l) {
        try {
          _layers.emplace_back(new TransformerDecoderLayer(model,
                                                           scope + "/layer_" + std::to_string(l),
                                                           with_encoder_attention));
        } catch (std::exception&) {
          if (l == 0)
            throw;
          else
            break;
        }
      }
    }

    void TransformerDecoder::set_vocabulary_mask(const StorageView& ids) {
      _proj.mask_weights(ids);
    }

    void TransformerDecoder::reset_vocabulary_mask() {
      _proj.reset_mask();
    }

    layers::DecoderState TransformerDecoder::initial_state() const {
      layers::DecoderState state;
      for (size_t i = 0; i < _layers.size(); ++i) {
        const std::string i_str = std::to_string(i);
        state.emplace("self_keys_" + i_str, StorageView(_device));
        state.emplace("self_values_" + i_str, StorageView(_device));
        if (_with_encoder_attention) {
          state.emplace("memory_keys_" + i_str, StorageView(_device));
          state.emplace("memory_values_" + i_str, StorageView(_device));
        }
      }
      return state;
    }

    bool TransformerDecoder::should_reorder_state(const std::string& name) const {
      // No need to reorder projected memory keys and values as they are the same for each beam.
      return !_with_encoder_attention || !starts_with(name, "memory");
    }

    void TransformerDecoder::operator()(dim_t step,
                                        const StorageView& ids,
                                        layers::DecoderState& state,
                                        StorageView* logits,
                                        StorageView* attention) {
      PROFILE("TransformerDecoder");
      StorageView layer_in(ids.device());
      StorageView layer_out(ids.device());

      _embeddings(ids, layer_in);
      if (_position_encoder)
        (*_position_encoder)(layer_in, step);

      const StorageView* memory = nullptr;
      const StorageView* memory_lengths = nullptr;
      if (_with_encoder_attention) {
        memory = &state.at("memory");
        memory_lengths = &state.at("memory_lengths");
      }

      for (size_t l = 0; l < _layers.size(); ++l) {
        const std::string l_str = std::to_string(l);
        (*_layers[l])(layer_in,
                      memory,
                      memory_lengths,
                      state.at("self_keys_" + l_str),
                      state.at("self_values_" + l_str),
                      _with_encoder_attention ? &state.at("memory_keys_" + l_str) : nullptr,
                      _with_encoder_attention ? &state.at("memory_values_" + l_str) : nullptr,
                      layer_out,
                      l + 1 == _layers.size() ? attention : nullptr);
        swap(layer_in, layer_out);
      }

      if (logits) {
        _output_norm(layer_in, layer_in);
        _proj(layer_in, *logits);
      }
    }

  }
}
