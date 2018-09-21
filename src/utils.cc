#include "ctranslate2/utils.h"

#ifdef WITH_MKL
#  include <mkl.h>
#endif

namespace ctranslate2 {

  void init(size_t num_threads) {
#ifdef WITH_MKL
    // Set "enhanced performance" VML mode (see
    // https://software.intel.com/en-us/mkl-developer-reference-c-vm-data-types-accuracy-modes-and-performance-tips)
    vmlSetMode(VML_EP);
    if (num_threads > 0) {
      set_num_threads(num_threads);
    }
#endif
  }

  void set_num_threads(size_t num_threads) {
#ifdef WITH_MKL
    mkl_set_num_threads(num_threads);
#endif
  }

  bool support_avx2() {
#ifdef WITH_MKL
    return mkl_cbwr_get_auto_branch() >= MKL_CBWR_AVX2;
#else
    throw std::runtime_error("Checking AVX2 support requires Intel MKL");
#endif
  }

}
