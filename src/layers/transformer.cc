#include "ctranslate2/layers/transformer.h"

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
                                                     const bool with_encoder_attention,
                                                     const bool pre_norm,
                                                     const ops::ActivationType activation_type)
      : _self_attention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm)
      , _encoder_attention(with_encoder_attention
                           ? std::make_unique<MultiHeadAttention>(model,
                                                                  scope + "/attention",
                                                                  num_heads,
                                                                  /*self_attention=*/false,
                                                                  pre_norm)
                           : nullptr)
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
                           const Embeddings& embeddings) {
      if (model.get_variable_if_exists(scope + "/encodings"))
        return std::make_unique<PositionEmbedding>(model, scope);
      else
        return std::make_unique<SinusoidalPositionEncoder>(embeddings.output_size(),
                                                           embeddings.output_type(),
                                                           model.device());
    }


    TransformerEncoder::TransformerEncoder(const models::Model& model,
                                           const std::string& scope,
                                           const size_t num_heads,
                                           const bool with_position_encoding,
                                           const bool pre_norm,
                                           const ops::ActivationType activation_type)
      : _embeddings(model, scope + "/embeddings")
      , _num_heads(num_heads)
      , _compute_type(model.effective_compute_type())
      , _position_encoder(with_position_encoding
                          ? build_position_encoder(model, scope + "/position_encodings", _embeddings)
                          : nullptr)
      , _output_norm(pre_norm
                     ? std::make_unique<LayerNorm>(model, scope + "/layer_norm")
                     : nullptr) {
      for (size_t l = 0;; ++l) {
        const std::string layer_scope = scope + "/layer_" + std::to_string(l);
        try {
          auto layer = std::make_unique<TransformerEncoderLayer>(model,
                                                                 layer_scope,
                                                                 num_heads,
                                                                 pre_norm,
                                                                 activation_type);
          _layers.emplace_back(std::move(layer));
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
      StorageView input(output.dtype(), output.device());
      _embeddings(ids, input);
      if (_position_encoder)
        (*_position_encoder)(input);

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
                                           const bool with_position_encoding,
                                           const bool with_encoder_attention,
                                           const bool pre_norm,
                                           const ops::ActivationType activation_type)
      : Decoder(model.device())
      , _with_encoder_attention(with_encoder_attention)
      , _num_heads(num_heads)
      , _compute_type(model.effective_compute_type())
      , _embeddings(model, scope + "/embeddings")
      , _position_encoder(with_position_encoding
                          ? build_position_encoder(model, scope + "/position_encodings", _embeddings)
                          : nullptr)
      , _output_norm(pre_norm
                     ? std::make_unique<LayerNorm>(model, scope + "/layer_norm")
                     : nullptr)
      , _proj(model, scope + "/projection") {
      for (size_t l = 0;; ++l) {
        const std::string layer_scope = scope + "/layer_" + std::to_string(l);
        try {
          auto layer = std::make_unique<TransformerDecoderLayer>(model,
                                                                 layer_scope,
                                                                 num_heads,
                                                                 with_encoder_attention,
                                                                 pre_norm,
                                                                 activation_type);
          _layers.emplace_back(std::move(layer));
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

    DecoderState TransformerDecoder::initial_state() const {
      const DataType dtype = output_type();
      DecoderState state;
      for (size_t i = 0; i < _layers.size(); ++i) {
        const std::string i_str = std::to_string(i);
        state.emplace("self_keys_" + i_str, StorageView(dtype, _device));
        state.emplace("self_values_" + i_str, StorageView(dtype, _device));
        if (_with_encoder_attention) {
          state.emplace("memory_keys_" + i_str, StorageView(dtype, _device));
          state.emplace("memory_values_" + i_str, StorageView(dtype, _device));
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
      if (_position_encoder)
        (*_position_encoder)(layer_in, std::max(step, dim_t(0)));

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
      std::unique_ptr<const Padder> memory_padder;
      std::unique_ptr<const StorageView> memory_lengths_mask;
      if (_with_encoder_attention) {
        const StorageView& memory_lengths = state.at("memory_lengths");
        memory_lengths_mask = std::make_unique<StorageView>(
          layers::MultiHeadAttention::prepare_length_mask(memory_lengths,
                                                          _num_heads,
                                                          max_time));
        auto it = state.find("memory");
        if (it != state.end()) {
          memory = &it->second;
          if (allow_padding_removal) {
            memory_padder = std::make_unique<Padder>(memory_lengths, memory->dim(1));
            memory_padder->remove_padding(*memory);
          }
        }
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
                      memory_lengths_mask.get(),
                      cached_self_attn_keys,
                      cached_self_attn_values,
                      cached_attn_keys,
                      cached_attn_values,
                      layer_out,
                      l + 1 == _layers.size() ? attention : nullptr,
                      input_padder.get(),
                      memory_padder.get());
        layer_in = std::move(layer_out);
      }

      if (step == 0) {
        // The memory is no longer needed as its projections were cached in the first step.
        state.erase("memory");
      }

      if (logits) {
        if (_output_norm)
          (*_output_norm)(layer_in, layer_in);
        _proj(layer_in, *logits);
      }

      if (input_padder)
        input_padder->add_padding(*logits);
    }

  }
}
