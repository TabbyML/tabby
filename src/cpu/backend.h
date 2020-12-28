#pragma once

#include <string>

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cpu {

    enum class GemmBackend {
      NONE,
      MKL,
      DNNL,
      ACCELERATE,
      OPENBLAS,
    };

    std::string gemm_backend_to_str(GemmBackend gemm_backend);
    bool mayiuse_mkl();
    GemmBackend get_gemm_backend(ComputeType compute_type);
    bool has_gemm_backend(ComputeType compute_type);
    bool prefer_u8s8s32_gemm();
    bool should_pack_gemm_weights();

  }
}
