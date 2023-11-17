#pragma once

#include "ctranslate2/layers/common.h"
#include "ctranslate2/padder.h"

namespace ctranslate2 {
  namespace layers {

    StorageView make_relative_positions(dim_t queries_length,
                                        dim_t keys_length,
                                        dim_t max_position);

    class RotaryEmbeddings;
    class Alibi;

    class MultiHeadAttention : public Layer
    {
    public:
      MultiHeadAttention(const models::Model& model,
                         const std::string& scope,
                         dim_t num_heads,
                         bool self_attention,
                         bool pre_norm = true,
                         bool is_decoder = false,
                         Alibi* alibi = nullptr);
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
                      const Padder* values_padder = nullptr,
                      bool return_normalized_attention = true,
                      StorageView* position_bias = nullptr,
                      dim_t offset = 0) const;

      bool has_positional_embeddings() const {
        return _relative_position_keys || _relative_attention_bias || _rotary_embeddings || _alibi;
      }

      bool multi_query() const {
        return _num_heads_kv == 1;
      }

      static StorageView prepare_length_mask(const StorageView& lengths,
                                             const dim_t num_heads,
                                             const dim_t num_queries,
                                             const bool mask_future = false,
                                             const bool multi_query = false);

    private:
      const dim_t _num_heads;
      const bool _self_attention;
      const bool _is_decoder;
      const std::vector<Dense> _linear;
      const dim_t _d_model;
      const dim_t _d_head;
      const bool _pre_norm;
      const std::unique_ptr<const LayerNorm> _layer_norm;
      const std::unique_ptr<RotaryEmbeddings> _rotary_embeddings;
      Alibi* _alibi;
      const StorageView* _relative_attention_bias;
      const StorageView* _relative_position_keys;
      const StorageView* _relative_position_values;
      dim_t _maximum_relative_position;
      const float _queries_scale;
      const dim_t _num_heads_kv;
      const bool _merge_time_and_head_dims;
      const dim_t _cache_time_dim;
      const dim_t _sliding_window;
    };

    enum class RotaryScalingType {
      None = -1,
      Linear,
    };

    class RotaryEmbeddings {
    public:
      RotaryEmbeddings(const dim_t dim = 0,
                       const bool interleave = true,
                       const RotaryScalingType scaling_type = RotaryScalingType::None,
                       const float scaling_factor = 1,
                       const float base = 10000,
                       const dim_t num_initial_positions = 2048);

      void apply(StorageView& x, const dim_t offset = 0);

    private:
      void initialize(const dim_t num_positions,
                      const dim_t dim,
                      const Device device,
                      const DataType dtype);

      const dim_t _dim;
      const bool _interleave;
      const RotaryScalingType _scaling_type;
      const float _scaling_factor;
      const float _base;
      const dim_t _num_initial_positions;
      const ops::Rotary _rotary_op;

      StorageView _sin;
      StorageView _cos;
    };


    class Alibi {
    public:
      Alibi(const bool use_positive_positions = false, const bool scale_alibi = false, const dim_t num_initial_positions = 2048);

      void apply(StorageView& x, const float scale = 1);

    private:
      const bool _use_positive_positions;
      const dim_t _num_initial_positions;
      const bool _scale_alibi;
      const ops::AlibiAdd _alibi_op;

      StorageView _alibi;
    };

  }
}
