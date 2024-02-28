#include "ctranslate2/ops/rms_norm.h"

#include <cub/block/block_reduce.cuh>

#include "cuda/helpers.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    constexpr dim_t num_threads = 512;

    template <typename T>
    __global__ void rms_norm_kernel(const T* input,
                                    const T* gamma,
                                    T* output,
                                    cuda::index_t depth,
                                    float epsilon,
                                    bool use_residual) {
      typedef cub::BlockReduce<float, num_threads> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      __shared__ float s_inv_rms;

      input += blockIdx.x * depth;
      output += blockIdx.x * depth;

      float sum_squares = 0;
      for (cuda::index_t i = threadIdx.x; i < depth; i += blockDim.x)
        sum_squares += float(input[i]) * float(input[i]);
      sum_squares = BlockReduce(temp_storage).Sum(sum_squares);

      if (threadIdx.x == 0)
        s_inv_rms = rsqrtf(sum_squares / depth + epsilon);

      __syncthreads();

      for (cuda::index_t i = threadIdx.x; i < depth; i += blockDim.x)
        if (use_residual)
          output[i] = float(input[i]) * s_inv_rms * (1 + float(gamma[i]));
        else
          output[i] = float(input[i]) * s_inv_rms * float(gamma[i]);
    }

    template <Device D, typename T>
    void RMSNorm::compute(const StorageView& gamma,
                          const StorageView& input,
                          StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      rms_norm_kernel<<<batch_size, num_threads, 0, cuda::get_cuda_stream()>>>(
        cuda::device_cast(input.data<T>()),
        cuda::device_cast(gamma.data<T>()),
        cuda::device_cast(output.data<T>()),
        depth,
        _epsilon,
        _use_residual);
    }

#define DECLARE_IMPL(T)                                                 \
    template void RMSNorm::compute<Device::CUDA, T>(const StorageView&, \
                                                    const StorageView&, \
                                                    StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
