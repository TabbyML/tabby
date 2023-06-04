#pragma once

#include <algorithm>

#ifdef _OPENMP
#  include <omp.h>
#else
#  include <BS_thread_pool_light.hpp>
#endif

#include "ctranslate2/types.h"
#include "ctranslate2/utils.h"

namespace ctranslate2 {
  namespace cpu {

    // The parallel_for construct is inspired by:
    // https://github.com/pytorch/pytorch/blob/cc3fc786b7ba04a0918f0e817a896a09f74f7e78/aten/src/ATen/ParallelOpenMP.h

    // Array smaller than this size will not be parallelized. This value could be smaller as
    // the number of computations per indices increases.
    constexpr dim_t GRAIN_SIZE = 32768;

    template <typename T>
    dim_t get_minimum_batch_copies_per_thread(const dim_t copy_size) {
      constexpr dim_t min_copy_bytes = 4096;
      const dim_t copy_bytes = copy_size * sizeof (T);
      return std::max(min_copy_bytes / copy_bytes, dim_t(1));
    }

#ifndef _OPENMP
    void set_num_threads(size_t num_threads);
    size_t get_num_threads();

    BS::thread_pool_light& get_thread_pool();
#endif

    template <typename Function>
    inline void parallel_for(const dim_t begin,
                             const dim_t end,
                             const dim_t grain_size,
                             const Function& f) {
      if (begin >= end) {
        return;
      }

      const dim_t size = end - begin;

#ifdef _OPENMP
      if (omp_get_max_threads() == 1 || omp_in_parallel() || size <= grain_size) {
        f(begin, end);
        return;
      }

      #pragma omp parallel
      {
        dim_t num_threads = omp_get_num_threads();
        if (grain_size > 0) {
          num_threads = std::min(num_threads, ceil_divide(size, grain_size));
        }

        const dim_t tid = omp_get_thread_num();
        const dim_t chunk_size = ceil_divide(size, num_threads);
        const dim_t begin_tid = begin + tid * chunk_size;
        if (begin_tid < end) {
          f(begin_tid, std::min(end, chunk_size + begin_tid));
        }
      }

#else
      dim_t num_blocks = get_num_threads();
      if (grain_size > 0) {
        num_blocks = std::min(num_blocks, ceil_divide(size, grain_size));
      }

      if (num_blocks == 1) {
        f(begin, end);
        return;
      }

      auto& thread_pool = get_thread_pool();
      thread_pool.push_loop(begin, end, f, num_blocks);
      thread_pool.wait_for_tasks();

#endif
    }

    template <typename T1, typename T2, typename Function>
    inline void unary_transform(const T1* x,
                                T2* y,
                                dim_t size,
                                const Function& func) {
      std::transform(x, x + size, y, func);
    }

    template <typename T1, typename T2, typename Function>
    inline void parallel_unary_transform(const T1* x,
                                         T2* y,
                                         dim_t size,
                                         dim_t work_size,
                                         const Function& func) {
      parallel_for(0, size, GRAIN_SIZE / work_size,
                   [x, y, &func](dim_t begin, dim_t end) {
                     std::transform(x + begin, x + end, y + begin, func);
                   });
    }

    template <typename T1, typename T2, typename T3, typename Function>
    inline void binary_transform(const T1* a,
                                 const T2* b,
                                 T3* c,
                                 dim_t size,
                                 const Function& func) {
      std::transform(a, a + size, b, c, func);
    }

    template <typename T1, typename T2, typename T3, typename Function>
    inline void parallel_binary_transform(const T1* a,
                                          const T2* b,
                                          T3* c,
                                          dim_t size,
                                          dim_t work_size,
                                          const Function& func) {
      parallel_for(0, size, GRAIN_SIZE / work_size,
                   [a, b, c, &func](dim_t begin, dim_t end) {
                     std::transform(a + begin, a + end, b + begin, c + begin, func);
                   });
    }

  }
}
