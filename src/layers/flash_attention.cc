#include "ctranslate2/layers/flash_attention.h"

namespace ctranslate2 {
  namespace layers {
    FlashMultiHeadAttention::FlashMultiHeadAttention(const models::Model& model,
                                           const std::string& scope,
                                           dim_t num_heads,
                                           bool self_attention,
                                           bool pre_norm,
                                           bool is_decoder,
                                           Alibi* alibi)
      : AttentionLayer(model, scope, num_heads, self_attention, pre_norm, is_decoder, alibi, true)
      , _cache_time_dim(1)
    {
      ERROR_CHECK((self_attention), "FlashAttention only supports the self-attention");
    }

    void FlashMultiHeadAttention::operator()(const StorageView& queries,
                                             const StorageView& values,
                                             const StorageView* values_lengths,
                                             StorageView& output,
                                             StorageView* cached_keys,
                                             StorageView* cached_values,
                                             StorageView* attention,
                                             const Padder* queries_padder,
                                             const Padder* values_padder,
                                             bool return_normalized_attention,
                                             StorageView* position_bias,
                                             dim_t offset) const {
      const Device device = queries.device();
      const DataType dtype = queries.dtype();

      StorageView fused_proj(dtype, device);
      StorageView queries_proj(dtype, device);
      StorageView keys_proj(dtype, device);
      StorageView values_proj(dtype, device);

      const StorageView* q = &queries;
      if (_layer_norm && _pre_norm) {
        (*_layer_norm)(queries, queries_proj);
        q = &queries_proj;
      }

      _linear[0](*q, fused_proj);

      dim_t beam_size = 1;

      bool prefilling = (_sliding_window > 0 && values_lengths);

      if (_num_heads_kv < _num_heads) {
        if (queries_padder)
          queries_padder->add_padding(fused_proj);

        const ops::Split split_op(2, {_d_model, _num_heads_kv * _d_head, _num_heads_kv * _d_head});
        split_op(fused_proj, queries_proj, keys_proj, values_proj);

        split_heads(queries_proj, _num_heads);
        split_heads(keys_proj, _num_heads_kv);
        split_heads(values_proj, _num_heads_kv);
      } else {
        split_heads(fused_proj, 3 * _num_heads, queries_padder);
        ops::Split(2)(fused_proj, queries_proj, keys_proj, values_proj);
      }

      if (_rotary_embeddings) {
        _rotary_embeddings->apply(queries_proj, offset, offset == 0);
        _rotary_embeddings->apply(keys_proj, offset, offset == 0);
      }

      if (cached_keys != nullptr) {
        if (cached_keys->empty()) {
          *cached_keys = std::move(keys_proj);
          *cached_values = std::move(values_proj);
        } else if (cached_keys->dim(_cache_time_dim) <= offset) {
          const ops::Concat concat_op(_cache_time_dim);
          auto shape = cached_keys->shape();
          shape[_cache_time_dim] = _offset_free_space;
          StorageView empty_storage(std::move(shape), dtype, device);
          StorageView& tmp = fused_proj;  // Reuse storage.
          tmp = std::move(*cached_keys);
          concat_op({&tmp, &empty_storage}, *cached_keys);
          tmp = std::move(*cached_values);
          concat_op({&tmp, &empty_storage}, *cached_values);

          if (!prefilling && _sliding_window > 0 && (offset / (_sliding_window - 1)) >= 1) {
            // only for generation
            const ops::Slide slide_op(_cache_time_dim, 1, cached_keys->shape()[_cache_time_dim] - 1);
            slide_op(*cached_keys, tmp);
            *cached_keys = std::move(tmp);
            slide_op(*cached_values, tmp);
            *cached_values = std::move(tmp);
          }
        }
      }

      if (cached_keys && offset == 0) {
        keys_proj.shallow_copy(*cached_keys);
        values_proj.shallow_copy(*cached_values);
      }

      StorageView* rotary_cos = nullptr;
      StorageView* rotary_sin = nullptr;
      bool rotary_interleaved = false;
      if (_rotary_embeddings && offset > 0) {
        rotary_cos = &(_rotary_embeddings->get_cos());
        rotary_sin = &(_rotary_embeddings->get_sin());
        rotary_interleaved = _rotary_embeddings->get_interleave();
      }

      // init output
      StorageView context(dtype, device);
      ops::FlashAttention fl_attn_ops(_queries_scale, _sliding_window);
      fl_attn_ops(queries_proj, keys_proj, values_proj, context, cached_keys, cached_values, attention,
                  return_normalized_attention, rotary_cos, rotary_sin, rotary_interleaved, nullptr/*alibli*/, offset);

      if (prefilling && cached_keys && cached_keys->shape()[_cache_time_dim] > _sliding_window) {
        // set only last sliding_window tokens to cached_keys and cached_values after computing attention
        const ops::Slide slide_op(_cache_time_dim, cached_keys->shape()[_cache_time_dim] - _sliding_window, _sliding_window);
        StorageView tmp(dtype, device);
        slide_op(*cached_keys, tmp);
        *cached_keys = std::move(tmp);
        slide_op(*cached_values, tmp);
        *cached_values = std::move(tmp);
      }
      combine_heads(context, _num_heads, queries_padder, beam_size);

      _linear.back()(context, output);
      if (_tensor_parallel) {
        StorageView tmp(output.shape(), output.dtype(), output.device());
        ops::ReduceAll ops_reduce_all(ops::ReduceAll::RED_OP::SUM);
        ops_reduce_all(output, tmp);
        output = std::move(tmp);
      }
      if (_layer_norm) {
        ops::Add()(queries, output, output);

        if (!_pre_norm)
          (*_layer_norm)(output, output);
      }
    }
    void FlashMultiHeadAttention::split_heads(StorageView& x,
                                              dim_t num_heads,
                                              const Padder* padder,
                                              dim_t beam_size) {
      if (padder)
        padder->add_padding(x);

      if (beam_size > 1)
        x.reshape({x.dim(0) / beam_size, beam_size, x.dim(2)});

      // x has shape [batch_size, time, depth]
      const dim_t batch_size = x.dim(0);
      const dim_t time = x.dim(1);
      const dim_t head_dim = x.dim(2) / num_heads;

      x.reshape({batch_size, time, num_heads, head_dim});
    }

    void FlashMultiHeadAttention::combine_heads(StorageView& x,
                                                dim_t num_heads,
                                                const Padder* padder,
                                                dim_t beam_size) {
      // x has shape [batch_size, num_heads, time, head_dim]
      const dim_t batch_size = x.dim(0);
      const dim_t time = x.dim(1);
      const dim_t depth = x.dim(3) * num_heads;

      x.reshape({batch_size, time, depth});

      if (beam_size > 1)
        x.reshape({batch_size * beam_size, 1, depth});

      if (padder)
        padder->remove_padding(x);
    }
  }
}