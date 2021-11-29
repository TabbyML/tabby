#include "ctranslate2/ops/mean.h"

#include <cub/cub.cuh>

#include "type_dispatch.h"
#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    constexpr dim_t num_threads = 256;

    template <typename T, typename AccumT>
    __global__ void mean_kernel(const T* input,
                                const cuda::index_t outer_size,
                                const cuda::index_t axis_size,
                                const cuda::index_t inner_size,
                                T* output) {
      typedef cub::BlockReduce<AccumT, num_threads> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;

      const cuda::index_t i = blockIdx.x / inner_size;
      const cuda::index_t j = blockIdx.x % inner_size;

      AccumT thread_sum = 0;
      for (cuda::index_t k = threadIdx.x; k < axis_size; k += blockDim.x) {
        thread_sum += AccumT(input[i * axis_size * inner_size + k * inner_size + j]);
      }

      AccumT sum = BlockReduce(temp_storage).Sum(thread_sum);

      if (threadIdx.x == 0) {
        output[blockIdx.x] = sum / AccumT(axis_size);
      }
    }

    template <Device D, typename T>
    void Mean::compute(const StorageView& input,
                       const dim_t outer_size,
                       const dim_t axis_size,
                       const dim_t inner_size,
                       StorageView& output) const {
      const dim_t blocks = std::min(outer_size * inner_size, cuda::max_blocks);
      mean_kernel<cuda::device_type<T>, float><<<blocks, num_threads, 0, cuda::get_cuda_stream()>>>(
        cuda::device_cast(input.data<T>()),
        outer_size,
        axis_size,
        inner_size,
        cuda::device_cast(output.data<T>()));
    }

#define DECLARE_IMPL(T)                                         \
    template void                                               \
    Mean::compute<Device::CUDA, T>(const StorageView& input,    \
                                   const dim_t outer_size,      \
                                   const dim_t axis_size,       \
                                   const dim_t inner_size,      \
                                   StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)

  }
}
