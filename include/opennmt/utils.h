#pragma once

#include <cstddef>
#include <cstdint>

#ifdef INTEL_MKL
#  include <mkl.h>
#endif

namespace opennmt
{

  inline void init(size_t num_threads) {
#ifdef WITH_MKL
    vmlSetMode(VML_EP);
    if (num_threads > 0)
      mkl_set_num_threads(num_threads);
#endif
  }

}
