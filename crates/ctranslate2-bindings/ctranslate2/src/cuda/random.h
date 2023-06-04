#pragma once

#include <curand_kernel.h>

namespace ctranslate2 {
  namespace cuda {

    curandStatePhilox4_32_10_t* get_curand_states(size_t num_states);

  }
}
