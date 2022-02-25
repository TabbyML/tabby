#include "ctranslate2/ops/dequantize.h"

#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    template <typename InT, typename OutT>
    struct dequantize_func {
      __device__ __forceinline__
      OutT operator()(float scale, InT x) const {
        return __fdividef(static_cast<float>(x), scale);
      }
    };

    template <Device D, typename InT, typename OutT>
    void Dequantize::dequantize(const StorageView& input,
                                const StorageView& scale,
                                StorageView& output) const {
      const dim_t depth = input.dim(-1);
      cuda::binary_transform(scale.data<float>(),
                             input.data<InT>(),
                             output.data<OutT>(),
                             input.size(),
                             dequantize_func<InT, cuda::device_type<OutT>>(),
                             cuda::repeat_vec_depth<dim_t>(depth));
    }

    template void
    Dequantize::dequantize<Device::CUDA, int8_t, float>(const StorageView&,
                                                        const StorageView&,
                                                        StorageView&) const;
    template void
    Dequantize::dequantize<Device::CUDA, int8_t, float16_t>(const StorageView&,
                                                            const StorageView&,
                                                            StorageView&) const;


    template <typename Epilogue, typename T>
    __global__ void dequantize_gemm_output_kernel(const int32_t* c,
                                                  const float* a_scales,
                                                  const float* b_scales,
                                                  const bool transpose_a,
                                                  const bool transpose_b,
                                                  const T* bias,
                                                  const Epilogue& epilogue,
                                                  T* y,
                                                  cuda::index_t depth) {
      // y = c / (expand_dims(a_scales, trans_a ? 0 : 1) * expand_dims(b_scales, trans_b ? 0 : 1)
      // if bias: y += expand_dims(bias, 0)
      // y = epilogue(y)
      const auto add_func = cuda::plus<T>();
      const auto rescale_func = dequantize_func<int32_t, T>();
      const cuda::index_t i = blockIdx.x;
      for (cuda::index_t j = threadIdx.x; j < depth; j += blockDim.x) {
        const cuda::index_t index = i * depth + j;
        const float scale = a_scales[transpose_a ? j : i] * b_scales[transpose_b ? j : i];
        T v = rescale_func(scale, c[index]);
        if (bias)
          v = add_func(v, bias[j]);
        y[index] = epilogue(v);
      }
    }

    template <typename T>
    static void dequantize_gemm_output_kernel_wrapper(const int32_t* c,
                                                      const float* a_scales,
                                                      const float* b_scales,
                                                      const bool transpose_a,
                                                      const bool transpose_b,
                                                      const T* bias,
                                                      const ActivationType* activation_type,
                                                      T* y,
                                                      dim_t batch_size,
                                                      dim_t depth) {
      const dim_t blocks = std::min(batch_size, cuda::max_blocks);
      const dim_t threads = std::min(depth, cuda::max_threads);

      if (!activation_type) {
        dequantize_gemm_output_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
          c, a_scales, b_scales, transpose_a, transpose_b, bias, thrust::identity<T>(), y, depth);

      } else {
        switch (*activation_type) {

        case ActivationType::ReLU: {
          dequantize_gemm_output_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
            c, a_scales, b_scales, transpose_a, transpose_b, bias, cuda::relu_func<T>(), y, depth);
          break;
        }

        case ActivationType::GELU: {
          dequantize_gemm_output_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
            c, a_scales, b_scales, transpose_a, transpose_b, bias, cuda::gelu_func<T>(), y, depth);
          break;
        }

        case ActivationType::Swish: {
          dequantize_gemm_output_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
            c, a_scales, b_scales, transpose_a, transpose_b, bias, cuda::swish_func<T>(), y, depth);
          break;
        }

        }
      }
    }

    template <Device D, typename T>
    void Dequantize::dequantize_gemm_output(const StorageView& c,
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
        bias ? cuda::device_cast<T>(bias->data<T>()) : nullptr,
        _activation_type,
        cuda::device_cast<T>(y.data<T>()),
        batch_size,
        depth);
    }

    template void
    Dequantize::dequantize_gemm_output<Device::CUDA, float>(const StorageView&,
                                                            const StorageView&,
                                                            const StorageView&,
                                                            const bool,
                                                            const bool,
                                                            const StorageView*,
                                                            StorageView&) const;
    template void
    Dequantize::dequantize_gemm_output<Device::CUDA, float16_t>(const StorageView&,
                                                                const StorageView&,
                                                                const StorageView&,
                                                                const bool,
                                                                const bool,
                                                                const StorageView*,
                                                                StorageView&) const;

  }
}
