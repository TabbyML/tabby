#pragma once

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cpu {

    enum class GemmBackend {
      NONE,
      MKL,
      DNNL,
    };

    bool mayiuse_mkl();
    GemmBackend get_gemm_backend(ComputeType compute_type);
    bool has_gemm_backend(ComputeType compute_type);
    bool prefer_u8s8s32_gemm();

  }
}
