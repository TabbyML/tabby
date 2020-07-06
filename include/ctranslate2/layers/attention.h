#pragma once

#include "ctranslate2/layers/common.h"

namespace ctranslate2 {
  namespace layers {

    StorageView make_relative_positions(dim_t length,
                                        dim_t max_position,
                                        bool with_cache = false);

    enum class LayerNormStrategy {
      Input,
      Output,
    };

    class MultiHeadAttention
    {
    public:
      MultiHeadAttention(const models::Model& model,
                         const std::string& scope,
                         dim_t num_heads,
                         bool self_attention,
                         LayerNormStrategy layer_norm_strategy = LayerNormStrategy::Input);
      void operator()(const StorageView& queries,
                      const StorageView* memory,
                      const StorageView* memory_lengths,
                      StorageView& output,
                      StorageView* cached_keys = nullptr,
                      StorageView* cached_values = nullptr,
                      StorageView* attention = nullptr) const;
    private:
      const dim_t _num_heads;
      const std::vector<Dense> _linear;
      const LayerNormStrategy _layer_norm_strategy;
      const LayerNorm _layer_norm;
      const StorageView* _relative_position_keys;
      const StorageView* _relative_position_values;
      const dim_t _maximum_relative_position;
      const ops::Transpose _transpose_op;

      void split_heads(const StorageView& x, StorageView& y) const;
      void combine_heads(const StorageView& x, StorageView& y) const;
    };

  }
}
