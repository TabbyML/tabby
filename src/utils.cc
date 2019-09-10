#include "ctranslate2/utils.h"

#ifdef WITH_MKL
#  include <mkl.h>
#  include <omp.h>
#endif

#ifdef WITH_CUDA
#  include "ctranslate2/cuda/utils.h"
#endif

namespace ctranslate2 {

  bool mayiuse_int16(Device device) {
    switch (device) {
#ifdef WITH_MKL
    case Device::CPU:
      return mkl_cbwr_get_auto_branch() >= MKL_CBWR_AVX2;
#endif
    default:
      return false;
    }
  }

  bool mayiuse_int8(Device device) {
    switch (device) {
#ifdef WITH_CUDA
    case Device::CUDA:
      return cuda::has_fast_int8();
#endif
#if defined(WITH_MKL) && defined(WITH_MKLDNN)
    case Device::CPU:
      return mkl_cbwr_get_auto_branch() >= MKL_CBWR_AVX2;
#endif
    default:
      return false;
    }
  }

  void set_num_threads(size_t num_threads) {
#ifdef WITH_MKL
    if (num_threads != 0)
      omp_set_num_threads(num_threads);
#endif
  }

}
