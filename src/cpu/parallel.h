#pragma once

#include <algorithm>

#ifdef _OPENMP
#  include <omp.h>
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

    template <typename Function>
    inline void parallel_for(const dim_t begin,
                             const dim_t end,
                             const dim_t grain_size,
                             const Function& f) {
      if (begin >= end) {
        return;
      }
#ifdef _OPENMP
      const dim_t size = end - begin;
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
      (void)grain_size;
      f(begin, end);
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
