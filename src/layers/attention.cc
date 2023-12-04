#include "ctranslate2/layers/attention.h"
#include "ctranslate2/ops/split.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "dispatch.h"
#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace layers {

    StorageView make_relative_positions(dim_t queries_length,
                                        dim_t keys_length,
                                        dim_t max_position) {
      StorageView positions({queries_length, keys_length}, DataType::INT32);
      auto* positions_data = positions.data<int32_t>();

      const dim_t offset = keys_length - queries_length;

      for (dim_t i = 0; i < queries_length; ++i) {
        auto* row = positions_data + i * keys_length;
        for (dim_t j = 0; j < keys_length; ++j) {
          row[j] = std::min(std::max(j - (i + offset), -max_position), max_position) + max_position;
        }
      }

      return positions;
    }

    static StorageView get_relative_position_bucket(bool bidirectional,
                                                    dim_t query_length,
                                                    dim_t key_length,
                                                    dim_t num_buckets,
                                                    dim_t max_distance,
                                                    dim_t query_offset = 0) {
      StorageView relative_buckets({query_length, key_length}, DataType::INT32);
      int32_t* relative_buckets_data = relative_buckets.data<int32_t>();

      if (bidirectional)
        num_buckets /= 2;

      const dim_t max_exact = num_buckets / 2;
      const float log_max_distance_over_max_exact = std::log(float(max_distance) / float(max_exact));

      cpu::parallel_for(0, query_length * key_length, 8, [&](const dim_t begin, const dim_t end) {
        for (dim_t flat_index = begin; flat_index < end; ++flat_index) {
          const dim_t i = flat_index / key_length;
          const dim_t j = flat_index % key_length;

          int32_t relative_position = j - (i + query_offset);
          int32_t relative_bucket = 0;

          if (bidirectional) {
            if (relative_position > 0)
              relative_bucket += num_buckets;
            else
              relative_position = std::abs(relative_position);
          } else {
            relative_position = -std::min(relative_position, 0);
          }

          const bool is_small = relative_position < max_exact;

          if (!is_small) {
            relative_position = std::min(
              int32_t(float(max_exact)
                      + std::log(float(relative_position) / float(max_exact))
                      / log_max_distance_over_max_exact
                      * float(num_buckets - max_exact)),
              int32_t(num_buckets - 1));
          }

          relative_bucket += relative_position;
          relative_buckets_data[flat_index] = relative_bucket;
        }
      });

      return relative_buckets;
    }

    static StorageView compute_relative_bias(const StorageView& relative_attention_bias,
                                             dim_t query_length,
                                             dim_t key_length,
                                             dim_t max_distance,
                                             bool is_decoder,
                                             dim_t query_offset = 0) {
      const Device device = relative_attention_bias.device();
      const DataType dtype = relative_attention_bias.dtype();
      const dim_t num_buckets = relative_attention_bias.dim(0);

      StorageView relative_attention_bucket = get_relative_position_bucket(!is_decoder,
                                                                           query_length,
                                                                           key_length,
                                                                           num_buckets,
                                                                           max_distance,
                                                                           query_offset);

      StorageView values(dtype, device);
      ops::Gather()(relative_attention_bias, relative_attention_bucket.to(device), values);

      StorageView values_t(dtype, device);
      ops::Transpose({2, 0, 1})(values, values_t);

      return values_t;
    }

    static StorageView build_alibi(dim_t num_heads,
                                   dim_t key_max_length,
                                   bool use_positive_positions,
                                   const float scale) {
      const float closest_power_of_2_f = std::pow(2.f, std::floor(std::log2f(num_heads)));
      const dim_t closest_power_of_2 = closest_power_of_2_f;

      const float base = std::pow(2.f, -std::pow(2.f, -(std::log2f(closest_power_of_2_f) - 3.f)));

      std::vector<float> slopes;
      slopes.reserve(closest_power_of_2);
      for (dim_t power = 1; power <= closest_power_of_2; ++power)
        slopes.emplace_back(std::pow(base, float(power)));

      if (closest_power_of_2 != num_heads) {
        const float extra_base = (
          std::pow(2.f, -std::pow(2.f, -(std::log2f(2 * closest_power_of_2_f) - 3.f))));
        const dim_t num_remaining_heads = std::min(
          closest_power_of_2, num_heads - closest_power_of_2);

        for (dim_t power = 1; power <= 2 * num_remaining_heads; power += 2)
          slopes.emplace_back(std::pow(extra_base, float(power)));
      }

      std::vector<float> positions(key_max_length);
      std::iota(positions.begin(),
                positions.end(),
                use_positive_positions ? 0 : -key_max_length + 1);

      StorageView alibi({1, num_heads, 1, key_max_length});

      for (dim_t h = 0; h < num_heads; ++h) {
        primitives<Device::CPU>::mul(slopes[h] * scale,
                                     positions.data(),
                                     alibi.index<float>({0, h, 0, 0}),
                                     key_max_length);
      }

      return alibi;
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

    static const ops::Transpose transpose_op({0, 2, 1, 3});

    static inline void save_attention(StorageView& attention, StorageView weights, dim_t beam_size) {
      if (beam_size == 1)
        attention = std::move(weights);
      else {
        transpose_op(weights, attention);
        attention.reshape({-1, weights.dim(1), 1, weights.dim(-1)});
      }
    }

    static void dot_product_attention(const StorageView& queries,
                                      const StorageView& keys,
                                      const StorageView& values,
                                      const StorageView* values_lengths,
                                      const StorageView* relative_position_keys,
                                      const StorageView* relative_position_values,
                                      const StorageView* relative_attention_bias,
                                      dim_t maximum_relative_position,
                                      StorageView& output,
                                      StorageView* attention = nullptr,
                                      bool return_normalized_attention = true,
                                      float queries_scale = 1,
                                      bool is_decoder = false,
                                      bool with_cache = false,
                                      dim_t beam_size = 1,
                                      Alibi* alibi = nullptr,
                                      StorageView* position_bias = nullptr) {
      PROFILE("dot_product_attention");

      std::unique_ptr<const StorageView> relative_positions;
      if (relative_position_keys || relative_position_values) {
        const dim_t query_length = queries.dim(2);
        const dim_t key_length = keys.dim(2);
        relative_positions = std::make_unique<StorageView>(
          make_relative_positions(query_length,
                                  key_length,
                                  maximum_relative_position).to(queries.device()));
      }

      const ops::MatMul keys_matmul(/*trans_a=*/false, /*trans_b=*/true, queries_scale);
      keys_matmul(queries, keys, output);
      if (relative_position_keys)
        add_relative_representations(queries,
                                     *relative_positions,
                                     *relative_position_keys,
                                     keys_matmul,
                                     output);

      if (relative_attention_bias) {
        StorageView local_position_bias(output.dtype(), output.device());

        if (!position_bias)
          position_bias = &local_position_bias;

        if (position_bias->empty()) {
          const dim_t query_length = queries.dim(2);
          const dim_t key_length = keys.dim(2);
          *position_bias = compute_relative_bias(*relative_attention_bias,
                                                 query_length,
                                                 key_length,
                                                 maximum_relative_position,
                                                 is_decoder,
                                                 with_cache ? key_length - 1 : 0);
        }

        DEVICE_AND_TYPE_DISPATCH(output.device(), output.dtype(),
                                 primitives<D>::add_batch_broadcast(position_bias->data<T>(),
                                                                    output.data<T>(),
                                                                    position_bias->size(),
                                                                    output.size()));
      }

      if (alibi)
        alibi->apply(output, queries_scale);

      StorageView attn(values.dtype(), values.device());
      ops::SoftMax()(output, values_lengths, attn);

      if (attention && !return_normalized_attention)
        save_attention(*attention, std::move(output), beam_size);

      const ops::MatMul values_matmul;
      values_matmul(attn, values, output);
      if (relative_position_values)
        add_relative_representations(attn,
                                     *relative_positions,
                                     *relative_position_values,
                                     values_matmul,
                                     output);

      if (attention && return_normalized_attention)
        save_attention(*attention, std::move(attn), beam_size);
    }

    static void split_heads(StorageView& x,
                            dim_t num_heads,
                            const Padder* padder = nullptr,
                            dim_t beam_size = 1) {
      if (padder)
        padder->add_padding(x);

      if (beam_size > 1)
        x.reshape({x.dim(0) / beam_size, beam_size, x.dim(2)});

      // x has shape [batch_size, time, depth]
      const dim_t batch_size = x.dim(0);
      const dim_t time = x.dim(1);
      const dim_t head_dim = x.dim(2) / num_heads;

      if (time == 1) {
        x.reshape({batch_size, num_heads, 1, head_dim});
      } else {
        x.reshape({batch_size, time, num_heads, head_dim});
        StorageView y(x.device(), x.dtype());
        transpose_op(x, y);
        x = std::move(y);
      }
    }

    static void replicate_heads(StorageView& x, dim_t repeats) {
      x.expand_dims(2);
      ops::Tile(2, repeats)(x);
      x.reshape({x.dim(0), x.dim(1) * x.dim(2), x.dim(3), x.dim(4)});
    }

    static void combine_heads(StorageView& x,
                              dim_t num_heads,
                              const Padder* padder = nullptr,
                              dim_t beam_size = 1) {
      // x has shape [batch_size, num_heads, time, head_dim]
      const dim_t batch_size = x.dim(0);
      const dim_t time = x.dim(2);
      const dim_t depth = x.dim(3) * num_heads;

      if (time > 1) {
        StorageView y(x.device(), x.dtype());
        transpose_op(x, y);
        x = std::move(y);
      }

      x.reshape({batch_size, time, depth});

      if (beam_size > 1)
        x.reshape({batch_size * beam_size, 1, depth});

      if (padder)
        padder->remove_padding(x);
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

    static std::unique_ptr<RotaryEmbeddings> make_rotary_embeddings(const models::Model& model,
                                                                    const std::string& scope) {
      const dim_t rotary_dim = model.get_attribute_with_default<int32_t>(scope + "/rotary_dim", -1);
      if (rotary_dim < 0)
        return nullptr;

      const bool interleave = model.get_flag_with_default(scope + "/rotary_interleave", true);
      const float base = model.get_attribute_with_default<float>(scope + "/rotary_base", 10000.f);

      const auto scaling_type = model.get_enum_value<RotaryScalingType>(
        scope + "/rotary_scaling_type", -1);
      const auto scaling_factor = model.get_attribute_with_default<float>(
        scope + "/rotary_scaling_factor", 1.f);

      return std::make_unique<RotaryEmbeddings>(rotary_dim,
                                                interleave,
                                                scaling_type,
                                                scaling_factor,
                                                base);
    }


    MultiHeadAttention::MultiHeadAttention(const models::Model& model,
                                           const std::string& scope,
                                           dim_t num_heads,
                                           bool self_attention,
                                           bool pre_norm,
                                           bool is_decoder,
                                           Alibi* alibi)
      : _num_heads(num_heads)
      , _self_attention(self_attention)
      , _is_decoder(is_decoder)
      , _linear(make_linear_layers(model, scope, self_attention))
      , _d_model(_linear.back().output_size())
      , _d_head(_d_model / _num_heads)
      , _pre_norm(pre_norm)
      , _layer_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _rotary_embeddings(make_rotary_embeddings(model, scope))
      , _alibi(alibi)
      , _relative_attention_bias(model.get_variable_if_exists(scope + "/relative_attention_bias"))
      , _relative_position_keys(model.get_variable_if_exists(scope + "/relative_position_keys"))
      , _relative_position_values(model.get_variable_if_exists(scope + "/relative_position_values"))
      , _queries_scale(model.get_attribute_with_default<float>(
                         scope + "/queries_scale",
                         1.f / std::sqrt(static_cast<float>(_d_head))))
      , _num_heads_kv(model.get_flag_with_default(scope + "/multi_query", false)
                      ? 1
                      : model.get_attribute_with_default<int32_t>(scope + "/num_heads_kv",
                                                                  _num_heads))
      , _merge_time_and_head_dims(_num_heads_kv == 1
                                  && !_relative_attention_bias
                                  && !_relative_position_keys
                                  && !_relative_position_values)
      , _cache_time_dim(_merge_time_and_head_dims ? 1 : 2)
      , _sliding_window(model.get_attribute_with_default<int32_t>(scope + "/sliding_window", 0))
    {
      if (_relative_position_keys)
        _maximum_relative_position = (_relative_position_keys->dim(0) - 1) / 2;
      else if (_relative_attention_bias)
        _maximum_relative_position = model.get_attribute<int32_t>(
          scope + "/relative_attention_max_distance");
      else
        _maximum_relative_position = 0;
    }

    DataType MultiHeadAttention::output_type() const {
      return _linear.back().output_type();
    }

    dim_t MultiHeadAttention::output_size() const {
      return _d_model;
    }

    void MultiHeadAttention::operator()(const StorageView& queries,
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
      PROFILE("MultiHeadAttention");
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

      if (!_self_attention) {
        queries_proj = std::move(fused_proj);

        if (cached_keys == nullptr || cached_keys->empty()) {
          _linear[1](values, fused_proj);

          if (_num_heads_kv == 1) {
            if (values_padder)
              values_padder->add_padding(fused_proj);
            ops::Split(2, {_d_head, _d_head})(fused_proj, keys_proj, values_proj);
          } else {
            split_heads(fused_proj, 2 * _num_heads, values_padder);
            ops::Split(1)(fused_proj, keys_proj, values_proj);
          }

          if (cached_keys != nullptr) {
            *cached_keys = std::move(keys_proj);
            *cached_values = std::move(values_proj);
          }
        }

        if (queries_proj.dim(1) == 1 && cached_keys)
          beam_size = queries_proj.dim(0) / cached_keys->dim(0);

        if (_num_heads_kv == 1) {
          if (queries_padder)
            queries_padder->add_padding(queries_proj);
          queries_proj.reshape({queries_proj.dim(0) / beam_size, -1, _d_head});
        } else {
          split_heads(queries_proj, _num_heads, queries_padder, beam_size);
        }

      } else {

        if (_num_heads_kv < _num_heads) {
          if (queries_padder)
            queries_padder->add_padding(fused_proj);

          const ops::Split split_op(2, {_d_model, _num_heads_kv * _d_head, _num_heads_kv * _d_head});
          split_op(fused_proj, queries_proj, keys_proj, values_proj);

          if (_merge_time_and_head_dims) {
            queries_proj.reshape({queries_proj.dim(0), -1, _d_head});
          } else {
            split_heads(queries_proj, _num_heads);
            split_heads(keys_proj, _num_heads_kv);
            split_heads(values_proj, _num_heads_kv);

            replicate_heads(keys_proj, _num_heads / _num_heads_kv);
            replicate_heads(values_proj, _num_heads / _num_heads_kv);
          }

        } else {
          split_heads(fused_proj, 3 * _num_heads, queries_padder);
          ops::Split(1)(fused_proj, queries_proj, keys_proj, values_proj);
        }

        if (_rotary_embeddings) {
          if (_merge_time_and_head_dims) {
            queries_proj.reshape({queries_proj.dim(0), -1, _d_model});
            split_heads(queries_proj, _num_heads);
          }

          _rotary_embeddings->apply(queries_proj, offset);
          _rotary_embeddings->apply(keys_proj, offset);

          if (_merge_time_and_head_dims) {
            combine_heads(queries_proj, _num_heads);
            queries_proj.reshape({queries_proj.dim(0), -1, _d_head});
          }
        }

        if (cached_keys != nullptr) {
          if (cached_keys->empty()) {
            *cached_keys = std::move(keys_proj);
            *cached_values = std::move(values_proj);
          } else {
            const ops::Concat concat_op(_cache_time_dim);
            StorageView& tmp = fused_proj;  // Reuse storage.
            tmp = std::move(*cached_keys);
            concat_op({&tmp, &keys_proj}, *cached_keys);
            tmp = std::move(*cached_values);
            concat_op({&tmp, &values_proj}, *cached_values);

            if (!prefilling && _sliding_window > 0 && cached_keys->shape()[2] > _sliding_window) {
              // only for generation
              const ops::Slide slide_op(2, 1, cached_keys->shape()[2] - 1);
              slide_op(*cached_keys, tmp);
              *cached_keys = std::move(tmp);
              slide_op(*cached_values, tmp);
              *cached_values = std::move(tmp);
            }
          }
        }
      }

      if (cached_keys) {
        keys_proj.shallow_copy(*cached_keys);
        values_proj.shallow_copy(*cached_values);
      }

      StorageView& context = fused_proj;  // Reuse storage.
      dot_product_attention(queries_proj,
                            keys_proj,
                            values_proj,
                            values_lengths,
                            _relative_position_keys,
                            _relative_position_values,
                            _relative_attention_bias,
                            _maximum_relative_position,
                            context,
                            attention,
                            return_normalized_attention,
                            _queries_scale,
                            _is_decoder,
                            bool(cached_keys),
                            beam_size,
                            _alibi,
                            position_bias);

      if (prefilling && cached_keys && cached_keys->shape()[2] > _sliding_window) {
        // set only last sliding_window tokens to cached_keys and cached_values after computing attention
        const ops::Slide slide_op(2, cached_keys->shape()[2] - _sliding_window, _sliding_window);
        StorageView tmp(dtype, device);
        slide_op(*cached_keys, tmp);
        *cached_keys = std::move(tmp);
        slide_op(*cached_values, tmp);
        *cached_values = std::move(tmp);
      }

      if (_merge_time_and_head_dims) {
        context.reshape(queries.shape());
        if (queries_padder)
          queries_padder->remove_padding(context);
      } else {
        combine_heads(context, _num_heads, queries_padder, beam_size);
      }

      _linear.back()(context, output);

      if (_layer_norm) {
        ops::Add()(queries, output, output);

        if (!_pre_norm)
          (*_layer_norm)(output, output);
      }
    }

    StorageView MultiHeadAttention::prepare_length_mask(const StorageView& lengths,
                                                        const dim_t num_heads,
                                                        const dim_t num_queries,
                                                        const bool mask_future,
                                                        const bool multi_query) {
      const Device device = lengths.device();
      const dim_t batch_size = lengths.size();
      StorageView mask(lengths.dtype(), device);

      if (multi_query)
        mask.resize({batch_size, num_queries, num_heads});
      else
        mask.resize({batch_size, num_heads, num_queries});

      DEVICE_DISPATCH(device, (primitives<D>::prepare_length_mask(lengths.data<int32_t>(),
                                                                  batch_size,
                                                                  num_heads,
                                                                  num_queries,
                                                                  mask_future,
                                                                  multi_query,
                                                                  mask.data<int32_t>())));
      return mask;
    }


    RotaryEmbeddings::RotaryEmbeddings(const dim_t dim,
                                       const bool interleave,
                                       const RotaryScalingType scaling_type,
                                       const float scaling_factor,
                                       const float base,
                                       const dim_t num_initial_positions)
      : _dim(dim)
      , _interleave(interleave)
      , _scaling_type(scaling_type)
      , _scaling_factor(scaling_factor)
      , _base(base)
      , _num_initial_positions(num_initial_positions)
      , _rotary_op(dim, interleave)
    {
    }

    void RotaryEmbeddings::apply(StorageView& x, const dim_t offset) {
      const Device device = x.device();
      const DataType dtype = x.dtype();
      const dim_t max_time = x.dim(-2);
      const dim_t dim = _dim == 0 ? x.dim(-1) : _dim;

      if (!_sin || offset + max_time > _sin.dim(0)) {
        const dim_t cur_num_positions = _sin ? _sin.dim(0) : 0;
        const dim_t new_num_positions = std::max(offset + max_time, cur_num_positions + _num_initial_positions);
        initialize(new_num_positions, dim, device, dtype);
      }

      StorageView sin(dtype, device);
      StorageView cos(dtype, device);
      TYPE_DISPATCH(dtype,
                    {
                      sin.view(_sin.index<T>({offset, 0}), {max_time, dim});
                      cos.view(_cos.index<T>({offset, 0}), {max_time, dim});
                    });

      StorageView y(dtype, device);
      _rotary_op(x, sin, cos, y);
      x = std::move(y);
    }

    void RotaryEmbeddings::initialize(const dim_t num_positions,
                                      const dim_t dim,
                                      const Device device,
                                      const DataType dtype) {
      StorageView inv_freq({1, dim / 2});
      for (dim_t i = 0; i < inv_freq.size(); ++i)
        inv_freq.at<float>(i) = 1.f / std::pow(_base, float(i * 2) / float(dim));
      if (inv_freq.device() != device)
        inv_freq = inv_freq.to(device);

      StorageView t({num_positions, 1});
      for (dim_t i = 0; i < t.size(); ++i)
        t.at<float>(i) = _scaling_type == RotaryScalingType::None ? i : float(i) / _scaling_factor;
      if (t.device() != device)
        t = t.to(device);

      StorageView freqs(device);
      ops::MatMul()(t, inv_freq, freqs);

      if (_interleave)
        freqs.expand_dims(-1);

      StorageView emb(device);
      ops::Concat(-1)({&freqs, &freqs}, emb);

      if (_interleave)
        emb.reshape({num_positions, dim});

      StorageView sin(device);
      ops::Sin()(emb, sin);
      if (sin.dtype() == dtype)
        _sin = std::move(sin);
      else
        _sin = sin.to(dtype);

      StorageView cos(device);
      ops::Cos()(emb, cos);
      if (cos.dtype() == dtype)
        _cos = std::move(cos);
      else
        _cos = cos.to(dtype);
    }


    Alibi::Alibi(const bool use_positive_positions, const bool scale_alibi, const dim_t num_initial_positions)
      : _use_positive_positions(use_positive_positions)
      , _num_initial_positions(num_initial_positions)
      , _scale_alibi(scale_alibi)
      , _alibi_op(use_positive_positions)
    {
    }

    void Alibi::apply(StorageView& x, const float scale) {
      const dim_t cur_length = _alibi ? _alibi.dim(-1) : 0;
      const dim_t key_length = x.dim(-1);

      if (key_length > cur_length) {
        const dim_t num_heads = x.dim(1);
        const dim_t new_length = cur_length + _num_initial_positions;
        _alibi = build_alibi(num_heads, new_length, _use_positive_positions, _scale_alibi ? scale : 1);
        _alibi.move_to(x.device(), x.dtype());
      }

      _alibi_op(x, _alibi, x);
    }

  }
}
