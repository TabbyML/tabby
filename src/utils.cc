#include "ctranslate2/utils.h"

#include <cstdlib>

#ifdef WITH_MKL
#  include <mkl.h>
#endif

namespace ctranslate2 {

  bool mayiuse_int16(Device device) {
#ifdef WITH_MKL
    return device == Device::CPU && mkl_cbwr_get_auto_branch() >= MKL_CBWR_AVX2;
#else
    return false;
#endif
  }

  bool mayiuse_int8(Device device) {
    const char* force_int8 = std::getenv("CT2_FORCE_INT8");
    if (!force_int8 || std::string(force_int8) != "1")
      return false;
    if (device == Device::CUDA)
      return true;
    else {
#if defined(WITH_MKL) && defined(WITH_MKLDNN)
      return mkl_cbwr_get_auto_branch() >= MKL_CBWR_AVX2;
#else
      return false;
#endif
    }
  }

}
