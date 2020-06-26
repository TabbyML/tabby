#pragma once

#ifdef _OPENMP
#  include <omp.h>
#endif

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cpu {

    template <typename T>
    inline T ceil_divide(const T& x, const T& y) {
      return (x + y - 1) / y;
    }

    // The parallel_for construct is inspired by:
    // https://github.com/pytorch/pytorch/blob/cc3fc786b7ba04a0918f0e817a896a09f74f7e78/aten/src/ATen/ParallelOpenMP.h

    // Array smaller than this size will not be parallelized. This value could be smaller as
    // the number of computations per indices increases.
    constexpr dim_t GRAIN_SIZE = 32768;

    template <typename Function>
    inline void parallel_for(const dim_t begin,
                             const dim_t end,
                             const dim_t work_size,
                             const Function& f) {
      if (begin >= end) {
        return;
      }
#ifdef _OPENMP
      const dim_t grain_size = GRAIN_SIZE / work_size;
      if (omp_get_max_threads() == 1 || omp_in_parallel() || (end - begin) <= grain_size) {
        f(begin, end);
        return;
      }
      #pragma omp parallel
      {
        dim_t num_threads = omp_get_num_threads();
        if (grain_size > 0) {
          num_threads = std::min(num_threads, ceil_divide((end - begin), grain_size));
        }

        const dim_t tid = omp_get_thread_num();
        const dim_t chunk_size = ceil_divide((end - begin), num_threads);
        const dim_t begin_tid = begin + tid * chunk_size;
        if (begin_tid < end) {
          f(begin_tid, std::min(end, chunk_size + begin_tid));
        }
      }
#else
      f(begin, end);
#endif
    }

  }
}
