#include "ctranslate2/ops/alibi_add.h"

#include "type_dispatch.h"
#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    __global__ void alibi_add_kernel(const T* input,
                                     const T* alibi,
                                     T* output,
                                     cuda::index_t alibi_offset,
                                     cuda::index_t num_heads,
                                     cuda::index_t query_length,
                                     cuda::index_t key_length,
                                     cuda::index_t cached_key_length) {
      const cuda::index_t i = blockIdx.x;
      const cuda::index_t h = (i / query_length) % num_heads;

      input += i * key_length;
      output += i * key_length;
      alibi += h * cached_key_length + alibi_offset;

      const cuda::plus<T> add;
      for (cuda::index_t j = threadIdx.x; j < key_length; j += blockDim.x)
        output[j] = add(input[j], alibi[j]);
    }

    template <Device D, typename T>
    void AlibiAdd::compute(const StorageView& input,
                           const StorageView& alibi,
                           const dim_t alibi_offset,
                           StorageView& output) const {
      const dim_t batch_size = input.dim(0);
      const dim_t num_heads = input.dim(1);
      const dim_t query_length = input.dim(2);
      const dim_t key_length = input.dim(3);

      const dim_t blocks = std::min(batch_size * num_heads * query_length, cuda::max_blocks);
      const dim_t threads = std::min(key_length, cuda::max_threads);

      alibi_add_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
        cuda::device_cast(input.data<T>()),
        cuda::device_cast(alibi.data<T>()),
        cuda::device_cast(output.data<T>()),
        alibi_offset,
        num_heads,
        query_length,
        key_length,
        alibi.dim(-1));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    AlibiAdd::compute<Device::CUDA, T>(const StorageView& input,        \
                                       const StorageView& alibi,        \
                                       const dim_t alibi_offset,        \
                                       StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
