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
      bool is_quantizable(const std::string& variable_name) const override;
      bool is_linear_weight(const std::string& variable_name) const override;
      bool is_packable(const std::string& variable_name) const override;
      void register_variable(const std::string& name, StorageView& variable) override;
      void finalize() override;

    private:
      size_t _num_heads;
      bool _with_relative_position;
      bool _pre_norm;
      ops::ActivationType _activation_type;
    };

  }
}
