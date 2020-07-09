#pragma once

#include "ctranslate2/ops/ops.h"
#include "ctranslate2/models/model.h"

namespace ctranslate2 {
  namespace layers {

    class Layer {
    public:
      virtual DataType output_type() const = 0;
    };

    class Embeddings : public Layer
    {
    public:
      Embeddings(const models::Model& model, const std::string& scope);
      DataType output_type() const override;
      void operator()(const StorageView& ids, StorageView& output) const;
    private:
      const ops::Gather _gather_op;
      const StorageView& _embeddings;
      const StorageView* _qscale;
      const std::unique_ptr<const StorageView> _scale;
    };

    class Dense : public Layer
    {
    public:
      Dense(const models::Model& model, const std::string& scope);
      DataType output_type() const override;
      void operator()(const StorageView& input, StorageView& output) const;
      void mask_weights(const StorageView& index);
      void reset_mask();
    private:
      bool _packed_weight;
      const StorageView& _weight;
      const StorageView* _bias;
      const StorageView* _qscale;
      const StorageView* _u8_shift_compensation;
      StorageView _partial_weight;
      StorageView _partial_bias;
      StorageView _partial_qscale;
      StorageView _partial_u8_shift_compensation;
      const ops::Gemm _gemm_op;
      const ops::Quantize _quantize_op;
      const ops::Dequantize _dequantize_op;
    };

    class LayerNorm : public Layer
    {
    public:
      LayerNorm(const models::Model& model, const std::string& scope);
      DataType output_type() const override;
      void operator()(const StorageView& input, StorageView& output) const;
    private:
      const ops::LayerNorm _norm_op;
      const StorageView& _beta;
      const StorageView& _gamma;
    };

  }
}
