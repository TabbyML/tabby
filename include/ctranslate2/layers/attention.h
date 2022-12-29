#pragma once

#include "ctranslate2/layers/common.h"
#include "ctranslate2/padder.h"

namespace ctranslate2 {
  namespace layers {

    StorageView make_relative_positions(dim_t length,
                                        dim_t max_position,
                                        bool with_cache = false);
    StorageView reduce_multi_head_attention(const StorageView& attention,
                                            dim_t num_heads_to_average);

    class MultiHeadAttention : public Layer
    {
    public:
      MultiHeadAttention(const models::Model& model,
                         const std::string& scope,
                         dim_t num_heads,
                         bool self_attention,
                         bool pre_norm = true,
                         bool is_decoder = false);
      DataType output_type() const override;
      dim_t output_size() const override;
      void operator()(const StorageView& queries,
                      const StorageView& values,
                      const StorageView* values_lengths,
                      StorageView& output,
                      StorageView* cached_keys = nullptr,
                      StorageView* cached_values = nullptr,
                      StorageView* attention = nullptr,
                      const Padder* queries_padder = nullptr,
                      const Padder* values_padder = nullptr) const;

      bool has_relative_position() const {
        return _relative_position_keys || _relative_attention_bias;
      }

      static StorageView prepare_length_mask(const StorageView& lengths,
                                             const dim_t num_heads,
                                             const dim_t num_queries,
                                             const bool mask_future = false);

    private:
      const dim_t _num_heads;
      const bool _self_attention;
      const bool _is_decoder;
      const std::vector<Dense> _linear;
      const dim_t _d_model;
      const bool _pre_norm;
      const LayerNorm _layer_norm;
      const StorageView* _relative_attention_bias;
      const StorageView* _relative_position_keys;
      const StorageView* _relative_position_values;
      dim_t _maximum_relative_position;
      const float _queries_scale;
    };

  }
}
