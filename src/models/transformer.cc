#include "ctranslate2/models/transformer.h"

#include "ctranslate2/layers/transformer.h"

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
      // Disallow packing for the last linear layer if it can be dynamically masked.
      return (is_linear_weight(variable_name)
              && (!get_vocabulary_map() || variable_name.find("projection") == std::string::npos));
    }

    void TransformerModel::register_variable(std::string name, StorageView variable) {
      if (_spec_revision == 1)
        name = map_v1_variable_name(std::move(name));
      SequenceToSequenceModel::register_variable(std::move(name), std::move(variable));
    }

    void TransformerModel::register_variable_alias(std::string alias, std::string variable_name) {
      if (_spec_revision == 1) {
        alias = map_v1_variable_name(std::move(alias));
        variable_name = map_v1_variable_name(std::move(variable_name));
      }
      SequenceToSequenceModel::register_variable_alias(std::move(alias), std::move(variable_name));
    }

    void TransformerModel::finalize() {
      SequenceToSequenceModel::finalize();
      if (_spec_revision >= 3)
        _num_heads = get_variable("num_heads").as_scalar<int8_t>();
      _with_relative_position = get_flag_with_default("with_relative_position", false);
      _pre_norm = get_flag_with_default("pre_norm", true);

      const auto* activation_type = get_variable_if_exists("activation");
      if (activation_type)
        _activation_type = static_cast<ops::ActivationType>(activation_type->as_scalar<int8_t>());
      else
        _activation_type = ops::ActivationType::ReLU;

      _alignment_layer = get_attribute_with_default<int16_t>("alignment_layer", -1);
      _alignment_heads = get_attribute_with_default<int16_t>("alignment_heads", 1);
    }

    std::unique_ptr<layers::Encoder> TransformerModel::make_encoder() const {
      return std::make_unique<layers::TransformerEncoder>(*this,
                                                          "encoder",
                                                          _num_heads,
                                                          !_with_relative_position,
                                                          _pre_norm,
                                                          _activation_type);
    }

    std::unique_ptr<layers::Decoder> TransformerModel::make_decoder() const {
      return std::make_unique<layers::TransformerDecoder>(*this,
                                                          "decoder",
                                                          _num_heads,
                                                          !_with_relative_position,
                                                          /*with_encoder_attention=*/true,
                                                          _pre_norm,
                                                          _activation_type,
                                                          _alignment_layer,
                                                          _alignment_heads);
    }

  }
}
