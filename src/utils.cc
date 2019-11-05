#include "ctranslate2/utils.h"

#ifdef WITH_MKL
#  include <mkl.h>
#endif

#ifdef _OPENMP
#  include <omp.h>
#endif

#ifdef WITH_CUDA
#  include "./cuda/utils.h"
#endif

namespace ctranslate2 {

#ifdef WITH_MKL
  static bool mkl_has_fast_int_gemm() {
#  if __INTEL_MKL__ > 2019 || (__INTEL_MKL__ == 2019 && __INTEL_MKL_UPDATE__ >= 5)
    // Intel MKL 2019.5 added optimized integers GEMM for SSE4.2 and AVX (in addition to
    // the existing AVX2 and AVX512), so it is virtually optimized for all target platforms.
    return true;
#  else
    return mkl_cbwr_get_auto_branch() >= MKL_CBWR_AVX2;
#  endif
  }
#endif

  bool mayiuse_int16(Device device) {
    switch (device) {
#ifdef WITH_MKL
    case Device::CPU:
      return mkl_has_fast_int_gemm();
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
      // Assume MKL-DNN was compiled against MKL otherwise it would only support AVX512.
      return mkl_has_fast_int_gemm();
#endif
    default:
      return false;
    }
  }

  void set_num_threads(size_t num_threads) {
#ifdef _OPENMP
    if (num_threads != 0)
      omp_set_num_threads(num_threads);
#endif
  }

  bool ends_with(const std::string& str, const std::string& suffix) {
    return (str.size() >= suffix.size() &&
            str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0);
  }

  bool starts_with(const std::string& str, const std::string& prefix) {
    return (str.size() >= prefix.size() &&
            str.compare(0, prefix.size(), prefix) == 0);
  }

}
