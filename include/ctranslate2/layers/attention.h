#pragma once

#include "ctranslate2/layers/common.h"

namespace ctranslate2 {
  namespace layers {

    class DotProductAttention
    {
    public:
      void operator()(const StorageView& queries,
                      const StorageView& keys,
                      const StorageView& values,
                      const StorageView* values_lengths,
                      StorageView& output,
                      StorageView* attention = nullptr,
                      float queries_scale = 1);
    };

    class MultiHeadAttention
    {
    public:
      MultiHeadAttention(const models::Model& model,
                         const std::string& scope,
                         dim_t num_heads,
                         bool self_attention);
      void operator()(const StorageView& queries,
                      const StorageView* memory,
                      const StorageView* memory_lengths,
                      StorageView& output,
                      StorageView* cached_keys = nullptr,
                      StorageView* cached_values = nullptr,
                      StorageView* attention = nullptr);
    private:
      dim_t _num_heads;
      std::vector<Dense> _linear;
      LayerNorm _layer_norm;
      DotProductAttention _attention;
      ops::Transpose _transpose_op;

      void split_heads(const StorageView& x, StorageView& y);
      void combine_heads(const StorageView& x, StorageView& y);
      static void cache_proj(StorageView& proj, StorageView& cache);
    };

  }
}
