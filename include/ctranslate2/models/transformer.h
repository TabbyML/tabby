#pragma once

// This file defines the execution engine for a TransformerSpec model.

#include "sequence_to_sequence.h"

#include "ctranslate2/ops/activation.h"

namespace ctranslate2 {
  namespace models {

    class TransformerModel : public SequenceToSequenceModel
    {
    public:
      TransformerModel(ModelReader& model_reader, size_t spec_revision, size_t num_heads = 0);
      size_t current_spec_revision() const override;
      std::unique_ptr<layers::Encoder> make_encoder() const override;
      std::unique_ptr<layers::Decoder> make_decoder() const override;

    protected:
      bool is_linear_weight(const std::string& variable_name) const override;
      bool is_packable(const std::string& variable_name) const override;
      void register_variable(std::string name, StorageView variable) override;
      void register_variable_alias(std::string alias, std::string variable_name) override;
      void initialize() override;

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

  }
}
