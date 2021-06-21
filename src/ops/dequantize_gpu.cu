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

    __global__ void dequantize_gemm_output_kernel(const int32_t* c,
                                                  const float* a_scales,
                                                  const float* b_scales,
                                                  const bool transpose_a,
                                                  const bool transpose_b,
                                                  const float* bias,
                                                  float* y,
                                                  dim_t depth) {
      // y = c / (expand_dims(a_scales, trans_a ? 0 : 1) * expand_dims(b_scales, trans_b ? 0 : 1)
      // if bias: y += expand_dims(bias, 0)
      const dim_t i = blockIdx.x;
      for (dim_t j = threadIdx.x; j < depth; j += blockDim.x) {
        const dim_t index = i * depth + j;
        y[index] = rescale(c[index],
                           a_scales[transpose_a ? j : i],
                           b_scales[transpose_b ? j : i]) + (bias ? bias[j] : 0);
      }
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
      const dim_t blocks = std::min(batch_size, cuda::max_blocks);
      const dim_t threads = std::min(depth, cuda::max_threads);
      dequantize_gemm_output_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
        c.data<int32_t>(),
        a_scale.data<float>(),
        b_scale.data<float>(),
        transpose_a,
        transpose_b,
        bias ? bias->data<float>() : nullptr,
        y.data<float>(),
        depth);
      if (_activation_type)
        get_activation_op(*_activation_type)(y, y);
    }

  }
}
