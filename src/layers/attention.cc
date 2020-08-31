#include "ctranslate2/layers/attention.h"

#include <algorithm>
#include <cmath>

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace layers {

    StorageView make_relative_positions(dim_t length, dim_t max_position, bool with_cache) {
      StorageView positions({with_cache ? 1 : length, length}, DataType::INT32);
      auto* positions_data = positions.data<int32_t>();

      if (with_cache) {
        for (dim_t i = 0; i < length; ++i) {
          positions_data[i] = std::max(i - length + 1, -max_position) + max_position;
        }
      } else {
        for (dim_t i = 0; i < length; ++i) {
          auto* row = positions_data + i * length;
          for (dim_t j = 0; j < length; ++j) {
            row[j] = std::min(std::max(j - i, -max_position), max_position) + max_position;
          }
        }
      }

      return positions;
    }

    static void matmul_with_relative_representations(const ops::MatMul& matmul_op,
                                                     const StorageView& a,
                                                     const StorageView& b,
                                                     StorageView& c) {
      const Device device = a.device();
      const DataType dtype = a.dtype();
      const dim_t batch = a.dim(0);
      const dim_t head = a.dim(1);
      const dim_t time = a.dim(2);

      StorageView a_t(dtype, device);
      ops::Transpose({2, 0, 1, 3})(a, a_t);
      a_t.reshape({time, batch * head, -1});

      StorageView c_t(dtype, device);
      matmul_op(a_t, b, c_t);
      c_t.reshape({time, batch, head, -1});
      ops::Transpose({1, 2, 0, 3})(c_t, c);
    }

    static void add_relative_representations(const StorageView& queries,
                                             const StorageView& relative_positions,
                                             const StorageView& relative_values,
                                             const ops::MatMul& matmul_op,
                                             StorageView& dot) {
      const Device device = queries.device();
      const DataType dtype = queries.dtype();

      StorageView relative_representations(dtype, device);
      ops::Gather()(relative_values, relative_positions, relative_representations);

      StorageView dot_relative(dtype, device);
      matmul_with_relative_representations(matmul_op,
                                           queries,
                                           relative_representations,
                                           dot_relative);
      ops::Add()(dot_relative, dot, dot);
    }

    static void dot_product_attention(const StorageView& queries,
                                      const StorageView& keys,
                                      const StorageView& values,
                                      const StorageView* values_lengths,
                                      const StorageView* relative_position_keys,
                                      const StorageView* relative_position_values,
                                      dim_t maximum_relative_position,
                                      StorageView& output,
                                      StorageView* attention = nullptr,
                                      float queries_scale = 1,
                                      bool with_cache = false) {
      PROFILE("dot_product_attention");

      std::unique_ptr<const StorageView> relative_positions;
      if (relative_position_keys || relative_position_values) {
        const dim_t max_time = keys.dim(2);
        relative_positions.reset(
          new StorageView(make_relative_positions(max_time,
                                                  maximum_relative_position,
                                                  with_cache).to(queries.device())));
      }

      const ops::MatMul keys_matmul(/*transpose_a=*/false, /*transpose_b=*/true, queries_scale);
      keys_matmul(queries, keys, output);
      if (relative_position_keys)
        add_relative_representations(queries,
                                     *relative_positions,
                                     *relative_position_keys,
                                     keys_matmul,
                                     output);

      StorageView attn(values.dtype(), values.device());
      ops::SoftMax()(output, values_lengths, attn);
      if (attention != nullptr) {
        // Transpose attn to make first head data contiguous.
        ops::Transpose({1, 0, 2, 3})(attn, output);
        attention->resize({attn.dim(0), attn.dim(2), attn.dim(3)});
        TYPE_DISPATCH(output.dtype(),
                      attention->copy_from(output.data<T>(),
                                           attention->size(),
                                           attention->device()));
      }

      const ops::MatMul values_matmul;
      values_matmul(attn, values, output);
      if (relative_position_values)
        add_relative_representations(attn,
                                     *relative_positions,
                                     *relative_position_values,
                                     values_matmul,
                                     output);
    }

    static std::vector<Dense> make_linear_layers(const models::Model& model,
                                                 const std::string& scope,
                                                 bool self_attention) {
      const dim_t num_linear_layers = self_attention ? 2 : 3;
      std::vector<Dense> layers;
      layers.reserve(num_linear_layers);
      for (dim_t i = 0; i < num_linear_layers; ++i)
        layers.emplace_back(model, scope + "/linear_" + std::to_string(i));
      return layers;
    }


    MultiHeadAttention::MultiHeadAttention(const models::Model& model,
                                           const std::string& scope,
                                           dim_t num_heads,
                                           bool self_attention,
                                           LayerNormStrategy layer_norm_strategy)
      : _num_heads(num_heads)
      , _self_attention(self_attention)
      , _linear(make_linear_layers(model, scope, self_attention))
      , _layer_norm_strategy(layer_norm_strategy)
      , _layer_norm(model, scope + "/layer_norm")
      , _relative_position_keys(model.get_variable_if_exists(scope + "/relative_position_keys"))
      , _relative_position_values(model.get_variable_if_exists(scope + "/relative_position_values"))
      , _maximum_relative_position(_relative_position_keys
                                   ? (_relative_position_keys->dim(0) - 1) / 2 : 0)
      , _queries_scale(1.f / std::sqrt(static_cast<float>(_layer_norm.output_size() / num_heads)))
      , _transpose_op({0, 2, 1, 3}) {
    }

    DataType MultiHeadAttention::output_type() const {
      return _layer_norm.output_type();
    }

    dim_t MultiHeadAttention::output_size() const {
      return _layer_norm.output_size();
    }

    void MultiHeadAttention::operator()(const StorageView& queries,
                                        const StorageView* memory,
                                        const StorageView* memory_lengths,
                                        StorageView& output,
                                        StorageView* cached_keys,
                                        StorageView* cached_values,
                                        StorageView* attention,
                                        const Padder* padder) const {
      PROFILE("MultiHeadAttention");
      const Device device = queries.device();
      const DataType dtype = queries.dtype();
      StorageView fused_proj(dtype, device);
      StorageView queries_proj(dtype, device);
      StorageView keys_proj(dtype, device);
      StorageView values_proj(dtype, device);
      StorageView split_queries(dtype, device);
      StorageView split_keys(dtype, device);
      StorageView split_values(dtype, device);

      if (_layer_norm_strategy == LayerNormStrategy::Input) {
        _layer_norm(queries, queries_proj);
        _linear[0](queries_proj, fused_proj);
      } else {
        _linear[0](queries, fused_proj);
      }

      if (!_self_attention) {
        split_heads(fused_proj, split_queries);
        if (cached_keys == nullptr || cached_keys->empty()) {
          _linear[1](*memory, fused_proj);
          ops::Split(-1)(fused_proj, keys_proj, values_proj);
          if (padder) {
            // From now on the time dimension is required.
            padder->add_padding(keys_proj);
            padder->add_padding(values_proj);
          }
          split_heads(keys_proj, split_keys);
          split_heads(values_proj, split_values);

          if (cached_keys != nullptr) {
            *cached_keys = std::move(split_keys);
            *cached_values = std::move(split_values);
            split_keys.shallow_copy(*cached_keys);
            split_values.shallow_copy(*cached_values);
          }
        } else {
          split_keys.shallow_copy(*cached_keys);
          split_values.shallow_copy(*cached_values);
        }
      } else {
        ops::Split(-1)(fused_proj, queries_proj, keys_proj, values_proj);
        if (padder) {
          // From now on the time dimension is required.
          padder->add_padding(queries_proj);
          padder->add_padding(keys_proj);
          padder->add_padding(values_proj);
        }
        split_heads(queries_proj, split_queries);
        split_heads(keys_proj, split_keys);
        split_heads(values_proj, split_values);

        if (cached_keys != nullptr) {
          if (cached_keys->empty()) {
            *cached_keys = std::move(split_keys);
            *cached_values = std::move(split_values);
          } else {
            StorageView& tmp = keys_proj;  // Reuse storage.
            tmp = std::move(*cached_keys);
            ops::Concat(2)({&tmp, &split_keys}, *cached_keys);
            tmp = std::move(*cached_values);
            ops::Concat(2)({&tmp, &split_values}, *cached_values);
          }
          split_keys.shallow_copy(*cached_keys);
          split_values.shallow_copy(*cached_values);
        }
      }

      StorageView& context = queries_proj;  // Reuse storage.
      dot_product_attention(split_queries,
                            split_keys,
                            split_values,
                            memory_lengths,
                            _relative_position_keys,
                            _relative_position_values,
                            _maximum_relative_position,
                            context,
                            attention,
                            _queries_scale,
                            bool(cached_keys));

      StorageView& combined = values_proj;  // Reuse storage.
      combine_heads(context, combined);

      if (padder && _self_attention) {
        // The time dimension is no longer needed.
        padder->remove_padding(combined);
      }

      _linear.back()(combined, output);
      ops::Add()(queries, output, output);
      if (_layer_norm_strategy == LayerNormStrategy::Output) {
        _layer_norm(output, output);
      }
    }

    void MultiHeadAttention::split_heads(StorageView& x, StorageView& y) const {
      const Shape original_shape = x.shape();
      x.reshape({x.dim(0), x.dim(1), _num_heads, x.dim(2) / _num_heads});
      _transpose_op(x, y);
      x.reshape(original_shape);
    }

    void MultiHeadAttention::combine_heads(const StorageView& x, StorageView& y) const {
      _transpose_op(x, y);
      y.reshape({y.dim(0), y.dim(1), y.dim(-1) * _num_heads});
    }

  }
}
