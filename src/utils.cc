#include "ctranslate2/utils.h"

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
