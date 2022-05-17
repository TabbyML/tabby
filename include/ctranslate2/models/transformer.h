#pragma once

#include "sequence_to_sequence.h"
#include "language_model.h"

#include "ctranslate2/ops/activation.h"

namespace ctranslate2 {
  namespace models {

    class TransformerModel : public SequenceToSequenceModel {
    public:
      TransformerModel(size_t num_heads = 0);
      size_t current_spec_revision() const override;
      std::unique_ptr<SequenceToSequenceReplica> as_sequence_to_sequence() const override;

    protected:
      bool is_linear_weight(const std::string& variable_name) const override;
      bool is_packable(const std::string& variable_name) const override;
      void register_variable(std::string name, StorageView variable) override;
      void register_variable_alias(std::string alias, std::string variable_name) override;
      void initialize(ModelReader& model_reader) override;

    private:
      size_t _num_heads;
      bool _with_relative_position;
      bool _pre_norm;
      ops::ActivationType _activation_type;
      dim_t _alignment_layer;
      dim_t _alignment_heads;
      layers::EmbeddingsMerge _embeddings_merge;
      bool _layernorm_embedding;
    };


    class TransformerDecoderModel : public LanguageModel {
    public:
      size_t current_spec_revision() const override;
      std::unique_ptr<SequenceGeneratorReplica> as_sequence_generator() const override;

    protected:
      bool is_linear_weight(const std::string& variable_name) const override;
      bool is_packable(const std::string& variable_name) const override;
      void initialize(ModelReader& model_reader) override;

    private:
      size_t _num_heads;
      bool _pre_norm;
      bool _no_final_norm;
      bool _layernorm_embedding;
      bool _project_in_out;
      ops::ActivationType _activation_type;
    };

  }
}
