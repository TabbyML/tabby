#pragma once

#include "cpu_generic.h"

namespace opennmt {
  namespace primitives {

    template<>
    void gemm(const int16_t* a, const int16_t* b,
              bool transpose_a, bool transpose_b,
              size_t m, size_t n, size_t k,
              int16_t alpha, int32_t beta,
              int32_t* c);

  }
}
