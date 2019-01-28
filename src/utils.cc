#include "ctranslate2/utils.h"

#ifdef WITH_MKL
#  include <mkl.h>
#else
#  include <stdexcept>
#endif

namespace ctranslate2 {

  bool support_avx2() {
#ifdef WITH_MKL
    return mkl_cbwr_get_auto_branch() >= MKL_CBWR_AVX2;
#else
    throw std::runtime_error("Checking AVX2 support requires Intel MKL");
#endif
  }

}
