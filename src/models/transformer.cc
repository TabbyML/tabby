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


    TransformerModel::TransformerModel(size_t num_heads)
      : _num_heads(num_heads) {
    }

    size_t TransformerModel::current_spec_revision() const {
      return 7;
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
      if (spec_revision() == 1)
        name = map_v1_variable_name(std::move(name));
      SequenceToSequenceModel::register_variable(std::move(name), std::move(variable));
    }

    void TransformerModel::initialize(ModelReader& model_reader) {
      SequenceToSequenceModel::initialize(model_reader);

      if (spec_revision() < 3) {
        register_variable("num_heads", StorageView(int8_t(_num_heads)));
      }

      if (spec_revision() < 5) {
        register_variable_alias("encoder/num_heads", "num_heads");
        register_variable_alias("encoder/pre_norm", "pre_norm");
        register_variable_alias("encoder/activation", "activation");
        register_variable_alias("encoder/embeddings_merge", "embeddings_merge");

        register_variable_alias("decoder/num_heads", "num_heads");
        register_variable_alias("decoder/pre_norm", "pre_norm");
        register_variable_alias("decoder/activation", "activation");
        register_variable_alias("decoder/alignment_layer", "alignment_layer");
        register_variable_alias("decoder/alignment_heads", "alignment_heads");
      }
    }

    std::unique_ptr<SequenceToSequenceReplica> TransformerModel::as_sequence_to_sequence() const {
      const auto scoped_device_setter = get_scoped_device_setter();

      auto encoder = std::make_unique<layers::TransformerEncoder>(*this, "encoder");
      auto decoder = std::make_unique<layers::TransformerDecoder>(*this, "decoder");

      const auto model = std::static_pointer_cast<const TransformerModel>(shared_from_this());
      return std::make_unique<EncoderDecoderReplica>(model, std::move(encoder), std::move(decoder));
    }

    std::unique_ptr<Model> TransformerModel::clone() const {
      return std::make_unique<TransformerModel>(*this);
    }


    size_t TransformerDecoderModel::current_spec_revision() const {
      return 8;
    }

    void TransformerDecoderModel::initialize(ModelReader& model_reader) {
      LanguageModel::initialize(model_reader);

      if (spec_revision() < 2) {
        register_variable_alias("decoder/num_heads", "num_heads");
        register_variable_alias("decoder/pre_norm", "pre_norm");
        register_variable_alias("decoder/activation", "activation");
      }
    }

    std::unique_ptr<SequenceGeneratorReplica>
    TransformerDecoderModel::as_sequence_generator() const {
      const auto scoped_device_setter = get_scoped_device_setter();

      auto decoder = std::make_unique<layers::TransformerDecoder>(*this, "decoder");

      const auto model = std::static_pointer_cast<const TransformerDecoderModel>(shared_from_this());
      return std::make_unique<DecoderReplica>(model, std::move(decoder));
    }

    bool TransformerDecoderModel::is_linear_weight(const std::string& variable_name) const {
      return is_quantizable(variable_name) && variable_name.find("embeddings") == std::string::npos;
    }

    std::unique_ptr<Model> TransformerDecoderModel::clone() const {
      return std::make_unique<TransformerDecoderModel>(*this);
    }


    size_t TransformerEncoderModel::current_spec_revision() const {
      return 1;
    }

    std::unique_ptr<SequenceEncoderReplica>
    TransformerEncoderModel::as_sequence_encoder() const {
      const auto scoped_device_setter = get_scoped_device_setter();

      auto encoder = std::make_unique<layers::TransformerEncoder>(*this, "encoder");

      const auto model = std::static_pointer_cast<const TransformerEncoderModel>(shared_from_this());
      return std::make_unique<EncoderReplica>(model, std::move(encoder));
    }

    bool TransformerEncoderModel::is_linear_weight(const std::string& variable_name) const {
      return is_quantizable(variable_name) && variable_name.find("embeddings") == std::string::npos;
    }

    std::unique_ptr<Model> TransformerEncoderModel::clone() const {
      return std::make_unique<TransformerEncoderModel>(*this);
    }

  }
}
