#pragma once

#include "ctranslate2/layers/common.h"

namespace ctranslate2 {
  namespace layers {

    StorageView make_relative_positions(dim_t length,
                                        dim_t max_position,
                                        bool with_cache = false);

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
      const StorageView* _relative_position_keys;
      const StorageView* _relative_position_values;
      const dim_t _maximum_relative_position;
      ops::Transpose _transpose_op;

      void split_heads(const StorageView& x, StorageView& y);
      void combine_heads(const StorageView& x, StorageView& y);
    };

  }
}
