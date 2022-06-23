#include "ctranslate2/layers/transformer.h"

#include <cmath>

namespace ctranslate2 {
  namespace layers {

    FeedForwardNetwork::FeedForwardNetwork(const models::Model& model,
                                           const std::string& scope,
                                           const bool pre_norm,
                                           const ops::ActivationType activation_type)
      : _layer_norm(model, scope + "/layer_norm")
      , _pre_norm(pre_norm)
      , _activation_type(activation_type)
      , _ff1(model, scope + "/linear_0", &_activation_type)
      , _ff2(model, scope + "/linear_1") {
    }

    void FeedForwardNetwork::operator()(const StorageView& input, StorageView& output) const {
      const StorageView* x = &input;
      if (_pre_norm) {
        _layer_norm(input, output);
        x = &output;
      }

      StorageView inner(input.dtype(), input.device());
      _ff1(*x, inner);
      _ff2(inner, output);
      ops::Add()(input, output, output);
      if (!_pre_norm)
        _layer_norm(output, output);
    }


    TransformerEncoderLayer::TransformerEncoderLayer(const models::Model& model,
                                                     const std::string& scope,
                                                     const size_t num_heads,
                                                     const bool pre_norm,
                                                     const ops::ActivationType activation_type)
      : _self_attention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm)
      , _ff(model, scope + "/ffn", pre_norm, activation_type) {
    }

    void TransformerEncoderLayer::operator()(const StorageView& input,
                                             const StorageView& lengths,
                                             StorageView& output,
                                             const Padder* padder) const {
      PROFILE("TransformerEncoderLayer");
      StorageView context(input.dtype(), input.device());
      _self_attention(input, input, &lengths, context, nullptr, nullptr, nullptr, padder, padder);
      _ff(context, output);
    }


    TransformerDecoderLayer::TransformerDecoderLayer(const models::Model& model,
                                                     const std::string& scope,
                                                     const size_t num_heads,
                                                     const bool pre_norm,
                                                     const ops::ActivationType activation_type)
      : _self_attention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm)
      , _encoder_attention(build_optional_layer<MultiHeadAttention>(model,
                                                                    scope + "/attention",
                                                                    num_heads,
                                                                    /*self_attention=*/false,
                                                                    pre_norm))
      , _ff(model, scope + "/ffn", pre_norm, activation_type) {
    }

    void TransformerDecoderLayer::operator()(const StorageView& input,
                                             const StorageView* input_length,
                                             const StorageView* memory,
                                             const StorageView* memory_lengths,
                                             StorageView* cached_self_attn_keys,
                                             StorageView* cached_self_attn_values,
                                             StorageView* cached_attn_keys,
                                             StorageView* cached_attn_values,
                                             StorageView& output,
                                             StorageView* attention,
                                             const Padder* input_padder,
                                             const Padder* memory_padder) const {
      PROFILE("TransformerDecoderLayer");
      _self_attention(input,
                      input,
                      input_length,
                      output,
                      cached_self_attn_keys,
                      cached_self_attn_values,
                      nullptr,
                      input_padder,
                      input_padder);

      StorageView context(input.dtype(), input.device());
      if (_encoder_attention) {
        (*_encoder_attention)(output,
                              *memory,
                              memory_lengths,
                              context,
                              cached_attn_keys,
                              cached_attn_values,
                              attention,
                              input_padder,
                              memory_padder);
      } else {
        context = std::move(output);
      }

      _ff(context, output);
    }


    static std::unique_ptr<PositionEncoder>
    build_position_encoder(const models::Model& model,
                           const std::string& scope,
                           const Layer& embeddings) {
      if (model.get_variable_if_exists(scope + "/encodings"))
        return std::make_unique<PositionEmbedding>(model, scope);
      else
        return std::make_unique<SinusoidalPositionEncoder>(embeddings.output_size(),
                                                           embeddings.output_type(),
                                                           model.device());
    }

    static std::unique_ptr<const StorageView>
    build_embeddings_scale(const models::Model& model,
                           const std::string& scope,
                           const Layer& embeddings) {
      const auto* scale = model.get_variable_if_exists(scope + "/scale_embeddings");

      // Backward compatibility with older models.
      if (!scale)
        scale = model.get_variable_if_exists(scope + "/embeddings/multiply_by_sqrt_depth");

      StorageView value;

      // The attribute can either be a boolean flag or the actual scale value.
      if (!scale || (scale->dtype() == DataType::INT8 && scale->as_scalar<int8_t>()))
        value = StorageView(std::sqrt(static_cast<float>(embeddings.output_size())));
      else if (scale->dtype() != DataType::INT8 && scale->as_scalar<float>() != 1.f)
        value = *scale;
      else
        return nullptr;

      return std::make_unique<StorageView>(value.to(embeddings.output_type()));
    }


    TransformerEncoder::TransformerEncoder(const models::Model& model,
                                           const std::string& scope,
                                           const size_t num_heads,
                                           const bool pre_norm,
                                           const ops::ActivationType activation_type,
                                           const EmbeddingsMerge embeddings_merge)
      : _embeddings(model, scope + "/embeddings", embeddings_merge)
      , _embeddings_scale(build_embeddings_scale(model, scope, _embeddings))
      , _num_heads(num_heads)
      , _compute_type(model.effective_compute_type())
      , _layernorm_embedding(build_optional_layer<LayerNorm>(model, scope + "/layernorm_embedding"))
      , _output_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _layers(build_layers_list<const TransformerEncoderLayer>(model,
                                                                 scope + "/layer",
                                                                 num_heads,
                                                                 pre_norm,
                                                                 activation_type))
      , _position_encoder(_layers.front()->has_relative_position()
                          ? nullptr
                          : build_position_encoder(model, scope + "/position_encodings", _embeddings))
    {
    }

    void TransformerEncoder::operator()(const std::vector<StorageView>& ids,
                                        const StorageView& lengths,
                                        StorageView& output) {
      PROFILE("TransformerEncoder");
      StorageView input(output.dtype(), output.device());
      _embeddings(ids, input);
      if (_embeddings_scale)
        ops::Mul()(input, *_embeddings_scale, input);
      if (_position_encoder)
        (*_position_encoder)(input);
      if (_layernorm_embedding)
        (*_layernorm_embedding)(input, input);

      const dim_t max_time = input.dim(1);

      // Remove padding to reduce the amount of computation.
      std::unique_ptr<Padder> padder;
      if (Padder::allow_padding_removal(output.device(), _compute_type)) {
        padder = std::make_unique<Padder>(lengths, max_time);
        padder->remove_padding(input);
      }

      const StorageView lengths_mask = layers::MultiHeadAttention::prepare_length_mask(lengths,
                                                                                       _num_heads,
                                                                                       max_time);

      for (size_t l = 0; l < _layers.size(); ++l) {
        (*_layers[l])(input, lengths_mask, output, padder.get());
        if (l + 1 < _layers.size())
          input = std::move(output);
      }
      if (_output_norm)
        (*_output_norm)(output, output);
      if (padder)
        padder->add_padding(output);
    }


    TransformerDecoder::TransformerDecoder(const models::Model& model,
                                           const std::string& scope,
                                           const size_t num_heads,
                                           const bool pre_norm,
                                           const ops::ActivationType activation_type,
                                           const dim_t alignment_layer,
                                           const dim_t alignment_heads)
      : Decoder(model.device())
      , _num_heads(num_heads)
      , _compute_type(model.effective_compute_type())
      , _embeddings(model, scope + "/embeddings")
      , _start_from_zero_embedding(model.get_flag_with_default(scope + "/start_from_zero_embedding",
                                                               false))
      , _embeddings_scale(build_embeddings_scale(model, scope, _embeddings))
      , _layernorm_embedding(build_optional_layer<LayerNorm>(model, scope + "/layernorm_embedding"))
      , _output_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _project_in(build_optional_layer<Dense>(model, scope + "/project_in"))
      , _project_out(build_optional_layer<Dense>(model, scope + "/project_out"))
      , _layers(build_layers_list<const TransformerDecoderLayer>(model,
                                                                 scope + "/layer",
                                                                 num_heads,
                                                                 pre_norm,
                                                                 activation_type))
      , _position_encoder(_layers.front()->has_relative_position()
                          ? nullptr
                          : build_position_encoder(model, scope + "/position_encodings", _embeddings))
      , _with_encoder_attention(_layers.front()->has_cross_attention())
      , _alignment_layer(alignment_layer < 0 ? _layers.size() + alignment_layer : alignment_layer)
      , _alignment_heads(alignment_heads == 0 ? _num_heads : alignment_heads)
      , _proj(model, scope + "/projection") {
    }

    DecoderState TransformerDecoder::initial_state(bool iterative_decoding) const {
      DecoderState state;
      if (iterative_decoding) {
        const DataType dtype = output_type();
        for (size_t i = 0; i < _layers.size(); ++i) {
          const std::string i_str = std::to_string(i);
          state.emplace("self_keys_" + i_str, StorageView(dtype, _device));
          state.emplace("self_values_" + i_str, StorageView(dtype, _device));
          if (_with_encoder_attention) {
            state.emplace("memory_keys_" + i_str, StorageView(dtype, _device));
            state.emplace("memory_values_" + i_str, StorageView(dtype, _device));
          }
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
                                        DecoderState& state,
                                        StorageView* logits,
                                        StorageView* attention) {
      return decode(ids, nullptr, step, state, logits, attention);
    }

    void TransformerDecoder::operator()(const StorageView& ids,
                                        const StorageView& lengths,
                                        DecoderState& state,
                                        StorageView& logits) {
      return decode(ids, &lengths, -1, state, &logits);
    }

    void TransformerDecoder::decode(const StorageView& ids,
                                    const StorageView* lengths,
                                    dim_t step,
                                    DecoderState& state,
                                    StorageView* logits,
                                    StorageView* attention) {
      PROFILE("TransformerDecoder");
      StorageView layer_in(output_type(), ids.device());
      StorageView layer_out(output_type(), ids.device());

      _embeddings(ids, layer_in);
      if (_start_from_zero_embedding)
        zero_first_timestep(layer_in, step);
      if (_embeddings_scale && (!_start_from_zero_embedding || step != 0))
        ops::Mul()(layer_in, *_embeddings_scale, layer_in);
      if (_project_in) {
        (*_project_in)(layer_in, layer_out);
        layer_in = std::move(layer_out);
      }
      if (layer_in.rank() == 2)
        layer_in.expand_dims(1);
      if (_position_encoder)
        (*_position_encoder)(layer_in, std::max(step, dim_t(0)));
      if (_layernorm_embedding)
        (*_layernorm_embedding)(layer_in, layer_in);

      const dim_t max_time = layer_in.dim(1);
      const bool allow_padding_removal = Padder::allow_padding_removal(_device, _compute_type);

      std::unique_ptr<const Padder> input_padder;
      std::unique_ptr<const StorageView> input_lengths_mask;
      if (lengths) {
        if (allow_padding_removal) {
          input_padder = std::make_unique<Padder>(*lengths, max_time);
          input_padder->remove_padding(layer_in);
        }
        input_lengths_mask = std::make_unique<StorageView>(
          layers::MultiHeadAttention::prepare_length_mask(*lengths,
                                                          _num_heads,
                                                          max_time,
                                                          /*mask_future=*/true));
      }

      StorageView* memory = nullptr;
      StorageView* memory_lengths_mask = nullptr;
      std::unique_ptr<const Padder> memory_padder;
      if (_with_encoder_attention) {
        if (step <= 0) {
          const StorageView& memory_lengths = state.at("memory_lengths");
          memory = &state.at("memory");
          if (allow_padding_removal) {
            memory_padder = std::make_unique<Padder>(memory_lengths, memory->dim(1));
            memory_padder->remove_padding(*memory);
          }

          state.emplace("memory_mask",
                        layers::MultiHeadAttention::prepare_length_mask(memory_lengths,
                                                                        _num_heads,
                                                                        max_time));
        }

        memory_lengths_mask = &state.at("memory_mask");
      }

      for (size_t l = 0; l < _layers.size(); ++l) {
        StorageView* cached_self_attn_keys = nullptr;
        StorageView* cached_self_attn_values = nullptr;
        StorageView* cached_attn_keys = nullptr;
        StorageView* cached_attn_values = nullptr;

        if (step >= 0) {
          const std::string l_str = std::to_string(l);
          cached_self_attn_keys = &state.at("self_keys_" + l_str);
          cached_self_attn_values = &state.at("self_values_" + l_str);
          if (_with_encoder_attention) {
            cached_attn_keys = &state.at("memory_keys_" + l_str);
            cached_attn_values = &state.at("memory_values_" + l_str);
          }
        }

        (*_layers[l])(layer_in,
                      input_lengths_mask.get(),
                      memory,
                      memory_lengths_mask,
                      cached_self_attn_keys,
                      cached_self_attn_values,
                      cached_attn_keys,
                      cached_attn_values,
                      layer_out,
                      l == size_t(_alignment_layer) ? attention : nullptr,
                      input_padder.get(),
                      memory_padder.get());
        layer_in = std::move(layer_out);
      }

      if (step == 0) {
        // The memory is no longer needed as its projections were cached in the first step.
        state.erase("memory");
        state.erase("memory_lengths");
      }

      if (attention) {
        *attention = reduce_multi_head_attention(*attention, _alignment_heads);
        if (step >= 0)
          attention->squeeze(1);
      }

      if (logits) {
        if (_output_norm)
          (*_output_norm)(layer_in, layer_in);
        if (_project_out) {
          (*_project_out)(layer_in, layer_out);
          layer_in = std::move(layer_out);
        }
        _proj(layer_in, *logits);

        if (step >= 0)
          logits->squeeze(1);
        else if (input_padder)
          input_padder->add_padding(*logits);
      }
    }

  }
}
