#include "ctranslate2/ops/multinomial.h"

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include "cuda/helpers.h"
#include "cuda/random.h"

namespace ctranslate2 {
  namespace ops {

    // Structure tracking the prefix sum of the previous block of threads.
    template <typename T>
    struct BlockPrefixSum {
      T prefix_sum = 0;

      __device__ T operator()(T block_aggregate) {
        T old_prefix = prefix_sum;
        prefix_sum += block_aggregate;
        return old_prefix;
      }
    };

    constexpr dim_t num_threads = 256;

    template <typename In, typename Out>
    __global__ void multinomial_kernel(const In* probs,
                                       cuda::index_t class_size,
                                       Out* output,
                                       curandStatePhilox4_32_10_t* states) {
      __shared__ float random_sample;
      if (threadIdx.x == 0)
        random_sample = curand_uniform(states + blockIdx.x);
      __syncthreads();

      typedef cub::BlockScan<float, num_threads> BlockScan;
      __shared__ typename BlockScan::TempStorage presum_temp_storage;

      BlockPrefixSum<float> prefix_op;
      Out candidate = class_size - 1;

      // In this loop we ensure that all threads do the same work,
      // even if some thread IDs are out of bounds.
      for (cuda::index_t offset = 0; offset < class_size; offset += blockDim.x) {
        const auto i = offset + threadIdx.x;
        float prob = i < class_size ? float(probs[blockIdx.x * class_size + i]) : 0.f;
        float prefix_sum_prob;
        BlockScan(presum_temp_storage).InclusiveSum(prob, prefix_sum_prob, prefix_op);
        __syncthreads();

        if (i < candidate && prefix_sum_prob >= random_sample)
          candidate = i;
      }

      // Get the first candidate.
      typedef cub::BlockReduce<Out, num_threads> BlockReduce;
      __shared__ typename BlockReduce::TempStorage min_temp_storage;
      Out first_candidate = BlockReduce(min_temp_storage).Reduce(candidate,
                                                                 cuda::minimum<Out>(),
                                                                 class_size);

      if (threadIdx.x == 0)
        output[blockIdx.x] = first_candidate;
    }

    template <Device D, typename T>
    void Multinomial::compute(const StorageView& input, StorageView& output) const {
      if (_sample_size != 1) {
        // The current CUDA kernel only returns a single sample per batch, so fallback on CPU.
        StorageView output_host(output.shape(), output.dtype());
        dispatch(input.to_float32().to(Device::CPU), output_host);
        output.copy_from(output_host);
        return;
      }

      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      const dim_t blocks = std::min(batch_size, cuda::max_blocks);

      // Get one curand state per block.
      auto* curand_states = cuda::get_curand_states(blocks);

      multinomial_kernel<<<blocks, num_threads, 0, cuda::get_cuda_stream()>>>(
        cuda::device_cast(input.data<T>()),
        depth,
        output.data<int32_t>(),
        curand_states);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Multinomial::compute<Device::CUDA, T>(const StorageView& input,     \
                                          StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
