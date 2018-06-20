#include "opennmt/utils.h"

#ifdef WITH_MKL
#  include <mkl.h>
#endif

namespace opennmt {

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
    mkl_set_num_threads(num_threads);
  }

  bool support_avx2() {
    return mkl_cbwr_get_auto_branch() >= MKL_CBWR_AVX2;
  }

}
