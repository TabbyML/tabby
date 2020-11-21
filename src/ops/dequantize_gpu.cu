#include "ctranslate2/ops/dequantize.h"

#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    struct dequantize_func {
      __device__
      float operator()(float scale, T x) {
        return __fdividef(static_cast<float>(x), scale);
      }
    };

    template<>
    void Dequantize::dequantize<Device::CUDA, int8_t>(const StorageView& input,
                                                      const StorageView& scale,
                                                      StorageView& output) const {
      const dim_t depth = input.dim(-1);
      cuda::binary_transform(scale.data<float>(),
                             input.data<int8_t>(),
                             output.data<float>(),
                             input.size(),
                             dequantize_func<int8_t>(),
                             cuda::repeat_vec_depth<dim_t>(depth));
    }


    __device__ __forceinline__ float rescale(const int32_t c,
                                             const float a_scale,
                                             const float b_scale) {
      return __fdividef(__int2float_rn(c), a_scale * b_scale);
    }

    struct rescale_func {
      __device__
      float operator()(int32_t x, const thrust::tuple<float, float>& scales) {
        return rescale(x, thrust::get<0>(scales), thrust::get<1>(scales));
      }
    };

    struct rescale_and_add_bias_func {
      __device__
      float operator()(int32_t x, const thrust::tuple<float, float, float>& args) {
        return rescale(x, thrust::get<0>(args), thrust::get<1>(args)) + thrust::get<2>(args);
      }
    };

    template <bool transpose_a, bool transpose_b>
    static inline void dequantize_gemm_output_kernel(const int32_t* c,
                                                     const float* a_scales,
                                                     const float* b_scales,
                                                     const float* bias,
                                                     float* y,
                                                     dim_t batch_size,
                                                     dim_t depth) {
#define EXPAND(scales, transpose)                                       \
      thrust::make_permutation_iterator(                                \
        scales,                                                         \
        thrust::make_transform_iterator(                                \
          thrust::counting_iterator<int>(0),                            \
          typename std::conditional<                                    \
            transpose,                                                  \
            cuda::repeat_vec<int>,                                      \
            cuda::repeat_vec_depth<int>>::type(depth)))

      // y = c / (expand_dims(a_scales, trans_a ? 0 : 1) * expand_dims(b_scales, trans_b ? 0 : 1)
      // if bias: y += expand_dims(bias, 0)
      auto a_scales_it = EXPAND(a_scales, transpose_a);
      auto b_scales_it = EXPAND(b_scales, transpose_b);
      const dim_t size = batch_size * depth;
      if (bias) {
        auto args = thrust::make_zip_iterator(thrust::make_tuple(a_scales_it,
                                                                 b_scales_it,
                                                                 EXPAND(bias, true)));
        THRUST_CALL(thrust::transform,
                    c, c + size,
                    args,
                    y,
                    rescale_and_add_bias_func());
      } else {
        auto scales_it = thrust::make_zip_iterator(thrust::make_tuple(a_scales_it, b_scales_it));
        THRUST_CALL(thrust::transform,
                    c, c + size,
                    scales_it,
                    y,
                    rescale_func());
      }

#undef EXPAND
    }

    template<>
    void Dequantize::dequantize_gemm_output<Device::CUDA>(const StorageView& c,
                                                          const StorageView& a_scale,
                                                          const StorageView& b_scale,
                                                          const bool transpose_a,
                                                          const bool transpose_b,
                                                          const StorageView* bias,
                                                          StorageView& y) const {
      const dim_t batch_size = a_scale.size();
      const dim_t depth = c.dim(-1);

      const auto* c_data = c.data<int32_t>();
      const auto* a_scale_data = a_scale.data<float>();
      const auto* b_scale_data = b_scale.data<float>();
      const auto* bias_data = bias ? bias->data<float>() : nullptr;
      auto* y_data = y.data<float>();

      if (transpose_a && transpose_b)
        dequantize_gemm_output_kernel<true, true>(c_data, a_scale_data, b_scale_data, bias_data,
                                                  y_data, batch_size, depth);
      else if (transpose_a)
        dequantize_gemm_output_kernel<true, false>(c_data, a_scale_data, b_scale_data, bias_data,
                                                   y_data, batch_size, depth);
      else if (transpose_b)
        dequantize_gemm_output_kernel<false, true>(c_data, a_scale_data, b_scale_data, bias_data,
                                                   y_data, batch_size, depth);
      else
        dequantize_gemm_output_kernel<false, false>(c_data, a_scale_data, b_scale_data, bias_data,
                                                    y_data, batch_size, depth);
    }

  }
}
