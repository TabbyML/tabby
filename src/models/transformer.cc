#include "ctranslate2/models/transformer.h"

#include "device_dispatch.h"
#include "type_dispatch.h"

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

    TransformerModel::TransformerModel(ModelReader& model_reader,
                                       size_t spec_revision,
                                       size_t num_heads)
      : SequenceToSequenceModel(model_reader, spec_revision)
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

    bool TransformerModel::is_quantizable(const std::string& variable_name) const {
      return ends_with(variable_name, "weight");
    }

    bool TransformerModel::is_linear_weight(const std::string& variable_name) const {
      // Linear weights are all variables that are quantizable and not under the "embeddings" scope.
      return is_quantizable(variable_name) && variable_name.find("embeddings") == std::string::npos;
    }

    bool TransformerModel::is_packable(const std::string& variable_name) const {
      // Disallow packing for the last linear layer which can be dynamically masked.
      return (is_linear_weight(variable_name)
              && variable_name.find("projection") == std::string::npos);
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


    static std::unique_ptr<layers::PositionEncoder>
    build_position_encoder(const TransformerModel& model,
                           const std::string& scope,
                           const layers::Embeddings& embeddings) {
      if (model.with_relative_position())
        return nullptr;
      if (model.get_variable_if_exists(scope + "/encodings"))
        return std::make_unique<layers::PositionEmbedding>(model, scope);
      else
        return std::make_unique<layers::SinusoidalPositionEncoder>(embeddings.output_size(),
                                                                   embeddings.output_type(),
                                                                   model.device());
    }


    TransformerFeedForward::TransformerFeedForward(const TransformerModel& model,
                                                   const std::string& scope)
      : _layer_norm(model, scope + "/layer_norm")
      , _activation(layers::ActivationType::ReLU)
      , _ff1(model, scope + "/linear_0", &_activation)
      , _ff2(model, scope + "/linear_1") {
    }

    void TransformerFeedForward::operator()(const StorageView& input, StorageView& output) const {
      StorageView inner(input.dtype(), input.device());
      _layer_norm(input, output);
      _ff1(output, inner);
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
                                             StorageView& output,
                                             const Padder* padder) const {
      PROFILE("TransformerEncoderLayer");
      StorageView context(input.dtype(), input.device());
      _self_attention(input, nullptr, &lengths, context, nullptr, nullptr, nullptr, padder);
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
                                             StorageView* attention,
                                             const Padder* padder) const {
      PROFILE("TransformerDecoderLayer");
      StorageView context(input.dtype(), input.device());
      if (_encoder_attention) {
        _self_attention(input, nullptr, nullptr, output,
                        &cached_self_attn_keys, &cached_self_attn_values);
        (*_encoder_attention)(output, memory, memory_lengths, context,
                              cached_attn_keys, cached_attn_values, attention, padder);
      } else {
        _self_attention(input, nullptr, nullptr, context,
                        &cached_self_attn_keys, &cached_self_attn_values);
      }
      _ff(context, output);
    }


    TransformerEncoder::TransformerEncoder(const TransformerModel& model, const std::string& scope)
      : _embeddings(model, scope + "/embeddings")
      , _compute_type(model.effective_compute_type())
      , _position_encoder(build_position_encoder(model, scope + "/position_encodings", _embeddings))
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

    DataType TransformerEncoder::output_type() const {
      return _output_norm.output_type();
    }

    dim_t TransformerEncoder::output_size() const {
      return _output_norm.output_size();
    }

    void TransformerEncoder::operator()(const StorageView& ids,
                                        const StorageView& lengths,
                                        StorageView& output) {
      PROFILE("TransformerEncoder");
      StorageView input(output.dtype(), output.device());
      _embeddings(ids, input);
      if (_position_encoder)
        (*_position_encoder)(input);

      // Remove padding to reduce the amount of computation.
      std::unique_ptr<Padder> padder;
      if (Padder::allow_padding_removal(output.device(), _compute_type)) {
        padder.reset(new Padder(lengths, input.dim(1)));
        padder->remove_padding(input);
      }

      for (size_t l = 0; l < _layers.size(); ++l) {
        (*_layers[l])(input, lengths, output, padder.get());
        if (l + 1 < _layers.size())
          input = std::move(output);
      }
      _output_norm(output, output);
      if (padder)
        padder->add_padding(output);
    }


    TransformerDecoder::TransformerDecoder(const TransformerModel& model,
                                           const std::string& scope,
                                           const bool with_encoder_attention)
      : Decoder(model.device())
      , _with_encoder_attention(with_encoder_attention)
      , _compute_type(model.effective_compute_type())
      , _embeddings(model, scope + "/embeddings")
      , _position_encoder(build_position_encoder(model, scope + "/position_encodings", _embeddings))
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

    DataType TransformerDecoder::output_type() const {
      return _proj.output_type();
    }

    dim_t TransformerDecoder::output_size() const {
      return _proj.output_size();
    }

    void TransformerDecoder::set_vocabulary_mask(const StorageView& ids) {
      _proj.mask_weights(ids);
    }

    void TransformerDecoder::reset_vocabulary_mask() {
      _proj.reset_mask();
    }

    layers::DecoderState TransformerDecoder::initial_state() const {
      const DataType dtype = output_type();
      layers::DecoderState state;
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
                                        layers::DecoderState& state,
                                        StorageView* logits,
                                        StorageView* attention) {
      PROFILE("TransformerDecoder");
      StorageView layer_in(output_type(), ids.device());
      StorageView layer_out(output_type(), ids.device());

      _embeddings(ids, layer_in);
      if (_position_encoder)
        (*_position_encoder)(layer_in, step);

      StorageView* memory = nullptr;
      const StorageView* memory_lengths = nullptr;
      std::unique_ptr<Padder> memory_padder;
      if (_with_encoder_attention) {
        memory_lengths = &state.at("memory_lengths");
        if (step == 0) {
          memory = &state.at("memory");
          if (Padder::allow_padding_removal(memory->device(), _compute_type)) {
            memory_padder.reset(new Padder(*memory_lengths, memory->dim(1)));
            memory_padder->remove_padding(*memory);
          }
        }
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
                      l + 1 == _layers.size() ? attention : nullptr,
                      memory_padder.get());
        layer_in = std::move(layer_out);
      }

      if (step == 0) {
        // The memory is no longer needed as its projections were cached in the first step.
        state.erase("memory");
      }

      if (logits) {
        _output_norm(layer_in, layer_in);
        _proj(layer_in, *logits);
      }
    }

  }
}
