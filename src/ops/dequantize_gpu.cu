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

    template <typename Epilogue>
    __global__ void dequantize_gemm_output_kernel(const int32_t* c,
                                                  const float* a_scales,
                                                  const float* b_scales,
                                                  const bool transpose_a,
                                                  const bool transpose_b,
                                                  const float* bias,
                                                  const Epilogue& epilogue,
                                                  float* y,
                                                  dim_t depth) {
      // y = c / (expand_dims(a_scales, trans_a ? 0 : 1) * expand_dims(b_scales, trans_b ? 0 : 1)
      // if bias: y += expand_dims(bias, 0)
      // y = epilogue(y)
      const dim_t i = blockIdx.x;
      for (dim_t j = threadIdx.x; j < depth; j += blockDim.x) {
        const dim_t index = i * depth + j;
        y[index] = epilogue(rescale(c[index],
                                    a_scales[transpose_a ? j : i],
                                    b_scales[transpose_b ? j : i]) + (bias ? bias[j] : 0));
      }
    }

    static void dequantize_gemm_output_kernel_wrapper(const int32_t* c,
                                                      const float* a_scales,
                                                      const float* b_scales,
                                                      const bool transpose_a,
                                                      const bool transpose_b,
                                                      const float* bias,
                                                      const ActivationType* activation_type,
                                                      float* y,
                                                      dim_t batch_size,
                                                      dim_t depth) {
      const dim_t blocks = std::min(batch_size, cuda::max_blocks);
      const dim_t threads = std::min(depth, cuda::max_threads);

      if (!activation_type) {
        dequantize_gemm_output_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
          c, a_scales, b_scales, transpose_a, transpose_b, bias, thrust::identity<float>(), y, depth);

      } else {
        switch (*activation_type) {

        case ActivationType::ReLU: {
          dequantize_gemm_output_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
            c, a_scales, b_scales, transpose_a, transpose_b, bias, cuda::relu_func<float>(), y, depth);
          break;
        }

        case ActivationType::GELU: {
          dequantize_gemm_output_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
            c, a_scales, b_scales, transpose_a, transpose_b, bias, cuda::gelu_func<float>(), y, depth);
          break;
        }

        }
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
      dequantize_gemm_output_kernel_wrapper(
        c.data<int32_t>(),
        a_scale.data<float>(),
        b_scale.data<float>(),
        transpose_a,
        transpose_b,
        bias ? bias->data<float>() : nullptr,
        _activation_type,
        y.data<float>(),
        batch_size,
        depth);
    }

  }
}
