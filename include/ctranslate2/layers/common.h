#pragma once

#include "ctranslate2/ops/ops.h"
#include "ctranslate2/models/model.h"

namespace ctranslate2 {
  namespace layers {

    class Embeddings
    {
    public:
      Embeddings(const models::Model& model, const std::string& scope);
      void operator()(const StorageView& ids, StorageView& output) const;
    private:
      const ops::Gather _gather_op;
      const StorageView& _embeddings;
      const StorageView* _qscale;
      const std::unique_ptr<const StorageView> _scale;
    };

    class Dense
    {
    public:
      Dense(const models::Model& model, const std::string& scope);
      void operator()(const StorageView& input, StorageView& output) const;
      void mask_weights(const StorageView& index);
      void reset_mask();
    private:
      bool _packed_weight;
      const StorageView& _weight;
      const StorageView* _bias;
      const StorageView* _qscale;
      const StorageView* _u8_shift_compensation;
      const float _u8_shift;
      StorageView _partial_weight;
      StorageView _partial_bias;
      StorageView _partial_qscale;
      StorageView _partial_u8_shift_compensation;
      const ops::Gemm _gemm_op;
    };

    class LayerNorm
    {
    public:
      LayerNorm(const models::Model& model, const std::string& scope);
      void operator()(const StorageView& input, StorageView& output) const;
    private:
      const ops::LayerNorm _norm_op;
      const StorageView& _beta;
      const StorageView& _gamma;
    };

  }
}
