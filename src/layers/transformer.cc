#include "ctranslate2/layers/transformer.h"

#include <cmath>

namespace ctranslate2 {
  namespace layers {

    FeedForwardNetwork::FeedForwardNetwork(const models::Model& model,
                                           const std::string& scope,
                                           const bool pre_norm,
                                           const ops::ActivationType activation_type)
      : _layer_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _pre_norm(pre_norm)
      , _activation_type(activation_type)
      , _ff1(model, scope + "/linear_0", &_activation_type)
      , _ff1_noact(build_optional_layer<Dense>(model, scope + "/linear_0_noact"))
      , _ff2(model, scope + "/linear_1") {
    }

    void FeedForwardNetwork::operator()(const StorageView& input, StorageView& output) const {
      const StorageView* x = &input;
      if (_layer_norm && _pre_norm) {
        (*_layer_norm)(input, output);
        x = &output;
      }

      const Device device = input.device();
      const DataType dtype = input.dtype();

      StorageView inner(dtype, device);
      _ff1(*x, inner);

      if (_ff1_noact) {
        StorageView linear(dtype, device);
        (*_ff1_noact)(*x, linear);
        ops::Mul()(linear, inner, inner);
      }

      _ff2(inner, output);

      if (_layer_norm) {
        ops::Add()(input, output, output);

        if (!_pre_norm)
          (*_layer_norm)(output, output);
      }
    }


    TransformerEncoderLayer::TransformerEncoderLayer(const models::Model& model,
                                                     const std::string& scope,
                                                     const dim_t num_heads,
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
                                             const StorageView* lengths,
                                             StorageView& output,
                                             const Padder* padder,
                                             StorageView* position_bias) const {
      PROFILE("TransformerEncoderLayer");
      StorageView context(input.dtype(), input.device());
      _self_attention(input,
                      input,
                      lengths,
                      context,
                      nullptr,
                      nullptr,
                      nullptr,
                      padder,
                      padder,
                      true,
                      position_bias);
      _ff(context, output);
    }


    TransformerDecoderLayer::TransformerDecoderLayer(const models::Model& model,
                                                     const std::string& scope,
                                                     const dim_t num_heads,
                                                     const bool pre_norm,
                                                     const ops::ActivationType activation_type,
                                                     Alibi* alibi)
      : _self_attention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm,
                        /*is_decoder=*/true,
                        alibi)
      , _shared_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/shared_layer_norm"))
      , _input_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/input_layer_norm"))
      , _post_attention_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/post_attention_layer_norm"))
      , _encoder_attention(build_optional_layer<MultiHeadAttention>(model,
                                                                    scope + "/attention",
                                                                    num_heads,
                                                                    /*self_attention=*/false,
                                                                    pre_norm,
                                                                    /*is_decoder=*/true))
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
                                             const Padder* memory_padder,
                                             bool return_normalized_attention,
                                             StorageView* position_bias,
                                             dim_t offset) const {
      PROFILE("TransformerDecoderLayer");

      const DataType dtype = input.dtype();
      const Device device = input.device();

      const bool use_parallel_residual = _shared_layer_norm || _input_layer_norm;

      if (use_parallel_residual) {
        // The parallel residual implementation assumes there is no cross attention.
        StorageView hidden(dtype, device);

        if (_shared_layer_norm)
          (*_shared_layer_norm)(input, hidden);
        else
          (*_input_layer_norm)(input, hidden);

        StorageView attn(dtype, device);
        _self_attention(hidden,
                        hidden,
                        input_length,
                        attn,
                        cached_self_attn_keys,
                        cached_self_attn_values,
                        nullptr,
                        input_padder,
                        input_padder,
                        true,
                        position_bias,
                        offset);

        if (_post_attention_layer_norm)
          (*_post_attention_layer_norm)(input, hidden);

        _ff(hidden, output);

        ops::Add()(output, input, output);
        ops::Add()(output, attn, output);

        return;
      }

      _self_attention(input,
                      input,
                      input_length,
                      output,
                      cached_self_attn_keys,
                      cached_self_attn_values,
                      nullptr,
                      input_padder,
                      input_padder,
                      true,
                      position_bias,
                      offset);

      StorageView context(dtype, device);
      if (_encoder_attention) {
        (*_encoder_attention)(output,
                              *memory,
                              memory_lengths,
                              context,
                              cached_attn_keys,
                              cached_attn_values,
                              attention,
                              input_padder,
                              memory_padder,
                              return_normalized_attention);
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


    TransformerEncoder::TransformerEncoder(const models::Model& model, const std::string& scope)
      : _embeddings(model, scope + "/embeddings",
                    model.get_enum_value<EmbeddingsMerge>(scope + "/embeddings_merge"))
      , _embeddings_scale(build_embeddings_scale(model, scope, _embeddings))
      , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 8))
      , _compute_type(model.effective_compute_type())
      , _layernorm_embedding(build_optional_layer<LayerNorm>(model, scope + "/layernorm_embedding"))
      , _output_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _layers(build_layers_list<const TransformerEncoderLayer>(
                  model,
                  scope + "/layer",
                  _num_heads,
                  model.get_flag_with_default(scope + "/pre_norm", true),
                  model.get_enum_value<ops::ActivationType>(scope + "/activation")))
      , _position_encoder(_layers.front()->get_self_attention().has_positional_embeddings()
                          ? nullptr
                          : build_position_encoder(model, scope + "/position_encodings", _embeddings))
    {
    }

    void TransformerEncoder::operator()(const std::vector<StorageView>& ids,
                                        const StorageView* lengths,
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
      std::unique_ptr<StorageView> lengths_mask;

      if (lengths) {
        if (Padder::allow_padding_removal(output.device(), _compute_type)) {
          padder = std::make_unique<Padder>(*lengths, max_time);
          padder->remove_padding(input);
        }

        lengths_mask = std::make_unique<StorageView>(
          layers::MultiHeadAttention::prepare_length_mask(*lengths, _num_heads, max_time));
      }

      StorageView position_bias(output.dtype(), output.device());

      for (size_t l = 0; l < _layers.size(); ++l) {
        (*_layers[l])(input, lengths_mask.get(), output, padder.get(), &position_bias);
        if (l + 1 < _layers.size())
          input = std::move(output);
      }
      if (_output_norm)
        (*_output_norm)(output, output);
      if (padder)
        padder->add_padding(output);
    }


    static std::unique_ptr<Alibi> make_alibi(const models::Model& model, const std::string& scope) {
      const bool use_alibi = model.get_flag_with_default(scope + "/alibi", false);
      if (!use_alibi)
        return nullptr;

      const bool use_positive_positions = model.get_flag_with_default(
        scope + "/alibi_use_positive_positions", true);
      const bool scale_alibi = model.get_flag_with_default(
        scope + "/scale_alibi", false);

      return std::make_unique<Alibi>(use_positive_positions, scale_alibi);
    }

    TransformerDecoder::TransformerDecoder(const models::Model& model, const std::string& scope)
      : Decoder(model.device())
      , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 8))
      , _compute_type(model.effective_compute_type())
      , _embeddings(model, scope + "/embeddings")
      , _start_from_zero_embedding(model.get_flag_with_default(scope + "/start_from_zero_embedding",
                                                               false))
      , _embeddings_scale(build_embeddings_scale(model, scope, _embeddings))
      , _layernorm_embedding(build_optional_layer<LayerNorm>(model, scope + "/layernorm_embedding"))
      , _output_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _project_in(build_optional_layer<Dense>(model, scope + "/project_in"))
      , _project_out(build_optional_layer<Dense>(model, scope + "/project_out"))
      , _alibi(make_alibi(model, scope))
      , _layers(build_layers_list<const TransformerDecoderLayer>(
                  model,
                  scope + "/layer",
                  _num_heads,
                  model.get_flag_with_default(scope + "/pre_norm", true),
                  model.get_enum_value<ops::ActivationType>(scope + "/activation"),
                  _alibi.get()))
      , _position_encoder(_layers.front()->get_self_attention().has_positional_embeddings()
                          ? nullptr
                          : build_position_encoder(model, scope + "/position_encodings", _embeddings))
      , _with_encoder_attention(_layers.front()->has_cross_attention())
      , _proj(model, scope + "/projection")
      , _sliding_window(model.get_attribute_with_default<int32_t>(scope + "/sliding_window", 0)) {

      dim_t alignment_layer = (
        model.get_attribute_with_default<int32_t>(scope + "/alignment_layer", -1));
      dim_t alignment_heads = (
        model.get_attribute_with_default<int32_t>(scope + "/alignment_heads", 1));

      if (alignment_layer < 0)
        alignment_layer = _layers.size() + alignment_layer;
      if (alignment_heads == 0)
        alignment_heads = _num_heads;

      set_alignment_heads(alignment_layer, alignment_heads);

      const auto* outputs_scale = model.get_variable_if_exists(scope + "/scale_outputs");
      if (outputs_scale) {
        const DataType dtype = get_default_float_type(_compute_type);
        _outputs_scale = std::make_unique<StorageView>(outputs_scale->to(dtype));
      }
    }

    DecoderState TransformerDecoder::initial_state(bool iterative_decoding) const {
      DecoderState state;

      if (iterative_decoding) {
        const size_t state_size = _layers.size() * (_with_encoder_attention ? 4 : 2);
        state.reserve(state_size);

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

    bool TransformerDecoder::replicate_state(const std::string& name) const {
      // No need to replicate projected memory keys and values as they are the same for each beam.
      return !_with_encoder_attention || !starts_with(name, "memory");
    }

    void TransformerDecoder::set_alignment_heads(const dim_t layer,
                                                 const dim_t num_heads_to_average) {
      std::vector<dim_t> range(num_heads_to_average);
      std::iota(range.begin(), range.end(), dim_t(0));

      _alignment_heads.clear();
      _alignment_heads.resize(_layers.size());
      _alignment_heads[layer] = std::move(range);

      _average_alignment_heads = true;
    }

    void TransformerDecoder::set_alignment_heads(const std::vector<std::pair<dim_t, dim_t>>& alignment_heads) {
      _alignment_heads.clear();
      _alignment_heads.resize(_layers.size());
      for (const auto& [layer, head] : alignment_heads)
        _alignment_heads[layer].push_back(head);

      _average_alignment_heads = false;
    }

    std::unique_ptr<StorageView>
    TransformerDecoder::get_layer_alignment_heads(const dim_t layer, const dim_t batch_size) const {
      if (_alignment_heads.empty())
        return nullptr;

      const auto& heads = _alignment_heads[layer];
      const dim_t num_heads = heads.size();

      if (heads.empty())
        return nullptr;

      std::vector<int32_t> indices;
      indices.reserve(batch_size * num_heads);
      for (dim_t i = 0; i < batch_size; ++i)
        indices.insert(indices.end(), heads.begin(), heads.end());

      return std::make_unique<StorageView>(Shape{batch_size, num_heads}, indices, _device);
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
                                        StorageView& logits,
                                        StorageView* attention) {
      return decode(ids, &lengths, -1, state, &logits, attention);
    }

    void TransformerDecoder::decode(const StorageView& ids,
                                    const StorageView* lengths,
                                    dim_t step,
                                    DecoderState& state,
                                    StorageView* outputs,
                                    StorageView* attention,
                                    bool return_logits) {
      PROFILE("TransformerDecoder");
      const DataType dtype = output_type();
      const Device device = ids.device();
      const bool is_sequence = ids.rank() > 1;

      StorageView layer_in(dtype, device);
      StorageView layer_out(dtype, device);

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

      const dim_t batch_size = layer_in.dim(0);
      dim_t max_time;

      if (_sliding_window > 0 && layer_in.dim(1) > _sliding_window) {
        max_time = _sliding_window;
      } else
        max_time = layer_in.dim(1);

      const bool allow_padding_removal = Padder::allow_padding_removal(_device, _compute_type);

      std::unique_ptr<const Padder> input_padder;
      std::unique_ptr<const StorageView> input_lengths;
      std::unique_ptr<const StorageView> input_lengths_mask;

      if (is_sequence && !lengths) {
        input_lengths = std::make_unique<StorageView>(Shape{ids.dim(0)}, int32_t(max_time), device);
        lengths = input_lengths.get();
      }

      bool multi_query = _layers.front()->get_self_attention().multi_query();

      if (lengths) {
        if (allow_padding_removal) {
          input_padder = std::make_unique<Padder>(*lengths, max_time);
          input_padder->remove_padding(layer_in);
        }

        StorageView lengths_mask = layers::MultiHeadAttention::prepare_length_mask(
          *lengths,
          _num_heads,
          max_time,
          /*mask_future=*/true,
          multi_query);

        if (step > 0)
          ops::Add()(lengths_mask, StorageView(int32_t(step)), lengths_mask);

        input_lengths_mask = std::make_unique<StorageView>(std::move(lengths_mask));
      }

      StorageView* memory = nullptr;
      std::unique_ptr<const StorageView> memory_lengths_mask;
      std::unique_ptr<const Padder> memory_padder;
      if (_with_encoder_attention) {
        const auto it = state.find("memory_lengths");
        const StorageView* memory_lengths = it != state.end() ? &it->second : nullptr;

        if (step <= 0) {
          memory = &state.at("memory");

          if (memory_lengths && allow_padding_removal) {
            memory_padder = std::make_unique<Padder>(*memory_lengths, memory->dim(1));
            memory_padder->remove_padding(*memory);
          }
        }

        if (memory_lengths) {
          const dim_t beam_size = batch_size / memory_lengths->dim(0);
          memory_lengths_mask = std::make_unique<StorageView>(
            layers::MultiHeadAttention::prepare_length_mask(*memory_lengths,
                                                            _num_heads,
                                                            beam_size > 1 ? beam_size : max_time));
        }
      }

      std::vector<StorageView> alignment_heads;
      if (attention)
        alignment_heads.reserve(_layers.size());

      StorageView position_bias(dtype, device);

      std::vector<StorageView> layer_ins;

      while (true) {
        dim_t prompt_size = layer_in.dim(1);
        if (_sliding_window == 0 || prompt_size <= _sliding_window) {
          layer_ins.push_back(std::move(layer_in));
          break;
        }
        if (layer_in.dim(1) > _sliding_window) {
          StorageView tmp(dtype, device);
          const ops::Split split_op(1, {_sliding_window, prompt_size - _sliding_window});
          split_op(layer_in, tmp, layer_in);
          layer_ins.push_back(std::move(tmp));
        }
      }

      for (size_t i = 0; i < layer_ins.size(); ++i) {
        auto layer_in_chunk = layer_ins[i];
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

          std::unique_ptr<StorageView> heads_to_select = get_layer_alignment_heads(l, batch_size);
          std::unique_ptr<StorageView> layer_attention;
          if (attention && heads_to_select)
            layer_attention = std::make_unique<StorageView>(dtype, device);

          dim_t offset = _sliding_window * i + step;
          offset = offset < 0 ? 0 : offset;
          if (i > 0) {
            auto max_tokens = _sliding_window + layer_in_chunk.dim(1);
            StorageView tmp_lengths = StorageView(Shape{layer_in_chunk.dim(0)}, int32_t(max_tokens), device);
            StorageView lengths_mask = layers::MultiHeadAttention::prepare_length_mask(
              tmp_lengths,
              _num_heads,
              max_tokens,
              /*mask_future=*/true,
              multi_query);

            const ops::Slide slide_lengths_op(2, _sliding_window, layer_in_chunk.dim(1));
            // reuse tmp_lengths
            slide_lengths_op(lengths_mask, tmp_lengths);
            input_lengths_mask = std::make_unique<StorageView>(std::move(tmp_lengths));
          }

          (*_layers[l])(layer_in_chunk,
                        input_lengths_mask.get(),
                        memory,
                        memory_lengths_mask.get(),
                        cached_self_attn_keys,
                        cached_self_attn_values,
                        cached_attn_keys,
                        cached_attn_values,
                        layer_out,
                        layer_attention.get(),
                        input_padder.get(),
                        memory_padder.get(),
                        return_normalized_attention(),
                        &position_bias,
                        offset);
          layer_in_chunk = std::move(layer_out);

          if (layer_attention) {
            alignment_heads.emplace_back(dtype, device);
            ops::Gather(1, 1)(*layer_attention, *heads_to_select, alignment_heads.back());
          }
        }
        layer_in = std::move(layer_in_chunk);
      }

      if (step == 0) {
        // The memory is no longer needed as its projections were cached in the first step.
        state.erase("memory");
      }

      if (attention && !alignment_heads.empty()) {
        if (_average_alignment_heads) {
          ops::Mean(1)(alignment_heads[0], *attention);
          if (!is_sequence)
            attention->squeeze(1);

        } else {
          std::vector<const StorageView*> alignment_heads_ptr;
          alignment_heads_ptr.reserve(alignment_heads.size());
          for (const auto& heads : alignment_heads)
            alignment_heads_ptr.emplace_back(&heads);

          ops::Concat(1)(alignment_heads_ptr, *attention);
          if (!is_sequence)
            attention->squeeze(2);
        }
      }

      if (outputs) {
        if (_output_norm)
          (*_output_norm)(layer_in, layer_in);
        if (_project_out) {
          (*_project_out)(layer_in, layer_out);
          layer_in = std::move(layer_out);
        }

        if (_outputs_scale)
          ops::Mul()(layer_in, *_outputs_scale, layer_in);

        if (return_logits)
          _proj(layer_in, *outputs);
        else
          *outputs = std::move(layer_in);

        if (!is_sequence)
          outputs->squeeze(1);
        else if (input_padder)
          input_padder->add_padding(*outputs);
      }
    }

  }
}
