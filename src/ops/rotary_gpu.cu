#include "ctranslate2/ops/rotary.h"

#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    struct ComputeType {
      using type = T;
    };

#if !CUDA_CAN_USE_HALF
    template<>
    struct ComputeType<__half> {
      using type = float;
    };
#endif

#if !CUDA_CAN_USE_BF16_MATH
    template<>
    struct ComputeType<__nv_bfloat16> {
      using type = float;
    };
#endif

    template <typename T, bool interleave>
    __global__ void rotary_kernel(const T* x,
                                  const T* sin,
                                  const T* cos,
                                  T* y,
                                  const cuda::index_t max_time,
                                  const cuda::index_t head_size,
                                  const cuda::index_t ndims,
                                  const cuda::index_t depth,
                                  const bool transpose) {
      const auto time = transpose ? blockIdx.x % max_time : blockIdx.x / head_size;
      const auto middle = ndims / 2;

      x += blockIdx.x * depth;
      y += blockIdx.x * depth;

      sin += time * ndims;
      cos += time * ndims;

      using C = typename ComputeType<T>::type;

      for (cuda::index_t i = threadIdx.x; i < depth; i += blockDim.x) {
        if (i >= ndims)
          y[i] = x[i];
        else if (interleave)
          y[i] = C(x[i]) * C(cos[i]) + (i % 2 == 0 ? -C(x[i + 1]) : C(x[i - 1])) * C(sin[i]);
        else
          y[i] = C(x[i]) * C(cos[i]) + (i < middle ? -C(x[i + middle]) : C(x[i - middle])) * C(sin[i]);
      }
    }

    template <Device D, typename T>
    void Rotary::compute(const StorageView& input,
                         const StorageView& sin,
                         const StorageView& cos,
                         StorageView& output,
                         bool is_transposed) const {
      const dim_t max_time = is_transposed ? input.dim(-2) : input.dim(-3);
      const dim_t head_size = is_transposed ? input.dim(-3) : input.dim(-2);
      const dim_t depth = input.dim(-1);
      const dim_t ndims = _ndims == 0 ? depth : _ndims;

      const dim_t blocks = std::min(input.size() / depth, cuda::max_blocks);
      const dim_t threads = std::min(depth, cuda::max_threads);

      const auto* x = cuda::device_cast(input.data<T>());
      const auto* s = cuda::device_cast(sin.data<T>());
      const auto* c = cuda::device_cast(cos.data<T>());
      auto* y = cuda::device_cast(output.data<T>());

      using DeviceT = cuda::device_type<T>;

      if (_interleave)
        rotary_kernel<DeviceT, true><<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
          x, s, c, y, max_time, head_size, ndims, depth, is_transposed);
      else
        rotary_kernel<DeviceT, false><<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
          x, s, c, y, max_time, head_size, ndims, depth, is_transposed);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Rotary::compute<Device::CUDA, T>(const StorageView&,                \
                                     const StorageView&,                \
                                     const StorageView&,                \
                                     StorageView&,                       \
                                     bool) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
