#include "ctranslate2/layers/attention.h"

#include <cmath>

namespace ctranslate2 {
  namespace layers {

    void DotProductAttention::operator()(const StorageView& queries,
                                         const StorageView& keys,
                                         const StorageView& values,
                                         const StorageView* values_lengths,
                                         StorageView& output,
                                         StorageView* attention,
                                         float queries_scale) {
      PROFILE("DotProductAttention");
      ops::MatMul(false, true, queries_scale)(queries, keys, output);

      StorageView attn(values.device());
      ops::SoftMax()(output, values_lengths, attn);
      if (attention != nullptr) {
        // Transpose attn to make first head data contiguous.
        ops::Transpose({1, 0, 2, 3})(attn, output);
        attention->resize({attn.dim(0), attn.dim(2), attn.dim(3)});
        attention->copy_from(output.data<float>(), attention->size(), attention->device());
      }

      ops::MatMul()(attn, values, output);
    }


    MultiHeadAttention::MultiHeadAttention(const models::Model& model,
                                           const std::string& scope,
                                           dim_t num_heads)
      : _num_heads(num_heads)
      , _layer_norm(model, scope + "/layer_norm")
      , _transpose_op({0, 2, 1, 3}) {
      for (size_t i = 0;; ++i) {
        try {
          _linear.emplace_back(model, scope + "/linear_" + std::to_string(i));
        } catch (std::exception&) {
          if (i == 0)
            throw;
          else
            break;
        }
      }
    }

    void MultiHeadAttention::operator()(const StorageView& queries,
                                        const StorageView* memory,
                                        const StorageView* memory_lengths,
                                        StorageView& output,
                                        StorageView* cached_keys,
                                        StorageView* cached_values,
                                        StorageView* attention) {
      PROFILE("MultiHeadAttention");
      Device device = queries.device();
      StorageView fused_proj(device);
      StorageView queries_proj(device);
      StorageView keys_proj(device);
      StorageView values_proj(device);
      StorageView split_queries(device);
      StorageView split_keys(device);
      StorageView split_values(device);

      _layer_norm(queries, queries_proj);
      _linear[0](queries_proj, fused_proj);

      if (memory) {
        split_heads(fused_proj, split_queries);
        if (cached_keys != nullptr && !cached_keys->empty()) {
          split_keys.shallow_copy(*cached_keys);
          split_values.shallow_copy(*cached_values);
        } else {
          _linear[1](*memory, fused_proj);
          ops::Split(-1)(fused_proj, keys_proj, values_proj);
          split_heads(keys_proj, split_keys);
          split_heads(values_proj, split_values);
          if (cached_keys != nullptr) {
            *cached_keys = split_keys;
            *cached_values = split_values;
          }
        }
      } else {
        ops::Split(-1)(fused_proj, queries_proj, keys_proj, values_proj);
        split_heads(queries_proj, split_queries);
        split_heads(keys_proj, split_keys);
        split_heads(values_proj, split_values);
        if (cached_keys != nullptr) {
          cache_proj(split_keys, *cached_keys);
          cache_proj(split_values, *cached_values);
          split_keys.shallow_copy(*cached_keys);
          split_values.shallow_copy(*cached_values);
        }
      }

      const dim_t dk = queries.dim(-1) / _num_heads;
      const float queries_scale = 1.0 / sqrt(dk);

      StorageView& context = queries_proj;  // Reuse storage.
      _attention(split_queries,
                 split_keys,
                 split_values,
                 memory_lengths,
                 context,
                 attention,
                 queries_scale);

      StorageView& combined = values_proj;  // Reuse storage.
      combine_heads(context, combined);

      _linear.back()(combined, output);
      ops::Add()(queries, output, output);
    }

    void MultiHeadAttention::split_heads(const StorageView& x, StorageView& y) {
      const StorageView z({x.dim(0), x.dim(1), _num_heads, x.dim(2) / _num_heads},
                          const_cast<float*>(x.data<float>()),
                          x.device());
      _transpose_op(z, y);
    }

    void MultiHeadAttention::combine_heads(const StorageView& x, StorageView& y) {
      _transpose_op(x, y);
      y.reshape({y.dim(0), y.dim(1), y.dim(-1) * _num_heads});
    }

    void MultiHeadAttention::cache_proj(StorageView& proj, StorageView& cache) {
      if (cache.empty()) {
        cache = proj;
      } else {
        StorageView tmp(proj.device());
        tmp = std::move(cache);
        ops::Concat(2)({&tmp, &proj}, cache);
      }
    }

  }
}
