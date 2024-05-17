#include "ctranslate2/layers/attention.h"
#include "ctranslate2/ops/split.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "dispatch.h"
#include "cpu/parallel.h"
#include <iostream>

namespace ctranslate2 {
  namespace layers {
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

    static std::vector<Dense> make_linear_layers(const models::Model& model,
                                                 const std::string& scope,
                                                 bool self_attention) {
      const dim_t num_linear_layers = self_attention ? 2 : 3;
      std::vector<Dense> layers;
      layers.reserve(num_linear_layers);
      for (dim_t i = 0; i < num_linear_layers; ++i)
        if (i == (num_linear_layers - 1)) {
          layers.emplace_back(model, scope + "/linear_" + std::to_string(i), nullptr, true);
        } else
          layers.emplace_back(model, scope + "/linear_" + std::to_string(i));
      return layers;
    }

    static std::unique_ptr<RotaryEmbeddings> make_rotary_embeddings(const models::Model& model,
                                                                    const std::string& scope,
                                                                    bool transpose) {
      const dim_t rotary_dim = model.get_attribute_with_default<int32_t>(scope + "/rotary_dim", -1);
      if (rotary_dim < 0)
        return nullptr;

      const bool interleave = model.get_flag_with_default(scope + "/rotary_interleave", true);
      const float base = model.get_attribute_with_default<float>(scope + "/rotary_base", 10000.f);

      const auto scaling_type = model.get_enum_value<RotaryScalingType>(
        scope + "/rotary_scaling_type", -1);
      const auto scaling_factor = model.get_attribute_with_default<float>(
        scope + "/rotary_scaling_factor", 1.f);
      const auto rotary_long_factor = model.get_variable_if_exists(scope +
                                                                        "/rotary_scaling_long_factor");
      const auto rotary_short_factor = model.get_variable_if_exists(scope +
                                                                   "/rotary_scaling_short_factor");
      const auto original_max_position_embeddings   = model.get_attribute_with_default<int32_t>(
        scope + "/original_max_position_embeddings", 0);

      const auto max_position_embeddings   = model.get_attribute_with_default<int32_t>(
        scope + "/max_position_embeddings", 0);

      return std::make_unique<RotaryEmbeddings>(rotary_dim,
                                                interleave,
                                                scaling_type,
                                                scaling_factor,
                                                base,
                                                /*num_initial_positions*/2048,
                                                rotary_long_factor,
                                                rotary_short_factor,
                                                original_max_position_embeddings,
                                                max_position_embeddings,
                                                transpose);
    }


    AttentionLayer::AttentionLayer(const models::Model& model,
                                           const std::string& scope,
                                           dim_t num_heads,
                                           bool self_attention,
                                           bool pre_norm,
                                           bool is_decoder,
                                           Alibi* alibi,
                                           bool is_flash_attn)
      : _tensor_parallel(model.tensor_parallel())
      , _num_heads(_tensor_parallel ? SAFE_DIVIDE(num_heads, ScopedMPISetter::getNRanks()) : num_heads)
      , _self_attention(self_attention)
      , _is_decoder(is_decoder)
      , _linear(make_linear_layers(model, scope, self_attention))
      , _d_model(_tensor_parallel ? SAFE_DIVIDE(_linear.back().output_size(),  ScopedMPISetter::getNRanks()) : _linear.back().output_size())
      , _d_head(model.get_attribute_with_default<int32_t >(scope + "/head_dim", _d_model / _num_heads))
      , _pre_norm(pre_norm)
      , _layer_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _rotary_embeddings(make_rotary_embeddings(model, scope, !is_flash_attn))
      , _alibi(alibi)
      , _queries_scale(model.get_attribute_with_default<float>(
                         scope + "/queries_scale",
                         1.f / std::sqrt(static_cast<float>(_d_head))))
      , _multi_query(model.get_flag_with_default(scope + "/multi_query", false))
      , _num_heads_kv(_multi_query
                      ? 1
                      : (_tensor_parallel ? model.get_attribute_with_default<int32_t>(scope + "/num_heads_kv",
                                            _num_heads * ScopedMPISetter::getNRanks()) / ScopedMPISetter::getNRanks()
                      : model.get_attribute_with_default<int32_t>(scope + "/num_heads_kv", _num_heads)))
      , _sliding_window(model.get_attribute_with_default<int32_t>(scope + "/sliding_window", 0))
    {
    }

    DataType AttentionLayer::output_type() const {
      return _linear.back().output_type();
    }

    dim_t AttentionLayer::output_size() const {
      return _d_model;
    }

    StorageView AttentionLayer::prepare_length_mask(const StorageView& lengths,
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
                                       const dim_t num_initial_positions,
                                       const StorageView* long_scaling_factor,
                                       const StorageView* short_scaling_factor,
                                       const dim_t original_max_position_embeddings,
                                       const dim_t max_position_embeddings,
                                       const bool transpose)
      : _dim(dim)
      , _interleave(interleave)
      , _scaling_type(scaling_type)
      , _scaling_factor(scaling_factor)
      , _base(base)
      , _num_initial_positions(num_initial_positions)
      , _rotary_scaling_long_factor(long_scaling_factor ?
                                    std::make_unique<StorageView>(*long_scaling_factor) : nullptr)
      , _rotary_scaling_short_factor(short_scaling_factor ?
                                    std::make_unique<StorageView>(*short_scaling_factor) : nullptr)
      , _original_max_position_embeddings(original_max_position_embeddings)
      , _max_position_embeddings(max_position_embeddings)
      , _rotary_op(dim, interleave)
      , _transpose(transpose)
    {
      if (_rotary_scaling_long_factor && _rotary_scaling_long_factor->device() != Device::CPU)
        _rotary_scaling_long_factor = std::make_unique<StorageView>(_rotary_scaling_long_factor->to(Device::CPU));
      if (_rotary_scaling_short_factor && _rotary_scaling_short_factor->device() != Device::CPU)
        _rotary_scaling_short_factor = std::make_unique<StorageView>(_rotary_scaling_short_factor->to(Device::CPU));
    }

    void RotaryEmbeddings::apply(StorageView& x, const dim_t offset, bool apply) {
      const Device device = x.device();
      const DataType dtype = x.dtype();
      const dim_t max_time = _transpose ? x.dim(-2) : x.dim(-3);
      const dim_t dim = _dim == 0 ? x.dim(-1) : _dim;

      if (!_sin || offset + max_time > _sin.dim(0)) {
        const dim_t cur_num_positions = _sin ? _sin.dim(0) : 0;
        const dim_t new_num_positions = std::max(offset + max_time, cur_num_positions + _num_initial_positions);
        initialize(new_num_positions, dim, device, dtype);
      }
      if (!apply)
        return;

      StorageView sin(dtype, device);
      StorageView cos(dtype, device);
      TYPE_DISPATCH(dtype,
                    {
                      sin.view(_sin.index<T>({offset, 0}), {max_time, dim});
                      cos.view(_cos.index<T>({offset, 0}), {max_time, dim});
                    });

      StorageView y(dtype, device);
      _rotary_op(x, sin, cos, y, _transpose);
      x = std::move(y);
    }

    void RotaryEmbeddings::initialize(const dim_t num_positions,
                                      const dim_t dim,
                                      const Device device,
                                      const DataType dtype) {
      StorageView inv_freq({1, dim / 2});
      if (_scaling_type == RotaryScalingType::Su) {
        StorageView* scaling_factor;
        for (dim_t i = 0; i < inv_freq.size(); ++i) {
          if (num_positions > _original_max_position_embeddings)
            scaling_factor = _rotary_scaling_long_factor.get();
          else
            scaling_factor = _rotary_scaling_short_factor.get();
          inv_freq.at<float>(i) = 1.f / (scaling_factor->at<float>(i) *
                                         (std::pow(_base, float(i * 2) / float(dim))));
        }
      }
      else {
        for (dim_t i = 0; i < inv_freq.size(); ++i)
          inv_freq.at<float>(i) = 1.f / std::pow(_base, float(i * 2) / float(dim));
      }
      if (inv_freq.device() != device)
        inv_freq = inv_freq.to(device);

      StorageView t({num_positions, 1});
      for (dim_t i = 0; i < t.size(); ++i)
        t.at<float>(i) = _scaling_type != RotaryScalingType::Linear ? i : float(i) / _scaling_factor;
      if (t.device() != device)
        t = t.to(device);

      StorageView freqs(device);
      ops::MatMul()(t, inv_freq, freqs);

      if (_interleave)
        freqs.expand_dims(-1);

      StorageView emb(device);
      ops::Concat(-1)({&freqs, &freqs}, emb);

      if (_interleave) {
        emb.reshape({num_positions, dim});
      }

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

      if (_original_max_position_embeddings != 0 && _max_position_embeddings != 0) {
        StorageView scaling_factor;
        float scale = _max_position_embeddings / _original_max_position_embeddings;
        if (scale <= 1)
          scaling_factor = StorageView(1.0f, device);
        else
          scaling_factor = StorageView(static_cast<float>(sqrt(1 + std::log(scale) / std::log(_original_max_position_embeddings))));

        ops::Mul()(_sin, scaling_factor, _sin);
        ops::Mul()(_cos, scaling_factor, _cos);
      }
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
