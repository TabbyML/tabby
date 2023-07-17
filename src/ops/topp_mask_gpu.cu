#include "ctranslate2/ops/topp_mask.h"

#include <cub/block/block_radix_sort.cuh>

#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    constexpr dim_t num_threads = 256;

    template <typename T, int ITEMS_PER_THREAD = 4>
    __global__ void topp_mask_kernel(const T* input,
                                     const T* probs,
                                     T* output,
                                     const float p,
                                     const float mask,
                                     const cuda::index_t class_size) {
      typedef cub::BlockRadixSort<float, num_threads, ITEMS_PER_THREAD, int> BlockRadixSort;
      typedef cub::BlockScan<float, num_threads> BlockScan;

      __shared__ union TempStorage {
        typename BlockRadixSort::TempStorage sort;
        typename BlockScan::TempStorage scan;
      } temp_storage;

      input += blockIdx.x * class_size;
      probs += blockIdx.x * class_size;
      output += blockIdx.x * class_size;

      float thread_keys[ITEMS_PER_THREAD];
      int thread_values[ITEMS_PER_THREAD];

      for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        const int id = threadIdx.x * ITEMS_PER_THREAD + i;

        if (id < class_size) {
          thread_keys[i] = probs[id];
          thread_values[i] = id;
        } else {
          thread_keys[i] = 0;
          thread_values[i] = -1;
        }
      }

      BlockRadixSort(temp_storage.sort).SortDescending(thread_keys, thread_values);

      __syncthreads();

      BlockScan(temp_storage.scan).ExclusiveSum(thread_keys, thread_keys);

      for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        const float total_p = thread_keys[i];
        const int id = thread_values[i];

        if (id < 0)
          break;

        output[id] = total_p < p ? input[id] : T(mask);
      }
    }

    static inline unsigned int next_power_of_2(unsigned int v) {
      v--;
      v |= v >> 1;
      v |= v >> 2;
      v |= v >> 4;
      v |= v >> 8;
      v |= v >> 16;
      v++;
      return v;
    }

    template <Device D, typename T>
    void TopPMask::compute(const StorageView& input,
                           const StorageView& probs,
                           StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      const dim_t blocks = std::min(batch_size, cuda::max_blocks);

      // TODO: support an arbitrary number of classes.

#define CASE_IPT(ITEMS_PER_THREAD)                                      \
      case ITEMS_PER_THREAD:                                            \
        topp_mask_kernel<cuda::device_type<T>, ITEMS_PER_THREAD>        \
          <<<blocks, num_threads, 0, cuda::get_cuda_stream()>>>(        \
          cuda::device_cast(input.data<T>()),                           \
          cuda::device_cast(probs.data<T>()),                           \
          cuda::device_cast(output.data<T>()),                          \
          _p,                                                           \
          _mask_value,                                                  \
          depth);                                                       \
        break

      const auto items_per_thread = next_power_of_2(ceil_divide(depth, num_threads));

      switch (items_per_thread) {
        CASE_IPT(1);
        CASE_IPT(2);
        CASE_IPT(4);
        CASE_IPT(8);
        CASE_IPT(16);
        CASE_IPT(32);

      default:
        throw std::runtime_error("The TopP operator does not support more than "
                                 + std::to_string(max_num_classes<Device::CUDA>())
                                 + " classes, but the input has "
                                 + std::to_string(depth) +
                                 " classes.");
      }

#undef CASE_IPT

    }

    template<>
    dim_t TopPMask::max_num_classes<Device::CUDA>() {
      return num_threads * 32;
    }

#define DECLARE_IMPL(T)                                                 \
    template void TopPMask::compute<Device::CUDA, T>(const StorageView&, \
                                                     const StorageView&, \
                                                     StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
