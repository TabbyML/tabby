#include "backend.h"

#include "ctranslate2/utils.h"
#include "cpu_info.h"
#include "env.h"

namespace ctranslate2 {
  namespace cpu {

    static bool mayiuse_mkl_init() {
      const std::string use_mkl_env = read_string_from_env("CT2_USE_MKL");
      if (use_mkl_env.empty()) {
#ifdef CT2_WITH_MKL
        return cpu_is_genuine_intel();
#else
        return false;
#endif
      } else {
        const bool use_mkl = string_to_bool(use_mkl_env);
#ifndef CT2_WITH_MKL
        if (use_mkl)
          throw std::invalid_argument("This CTranslate2 binary was not compiled with Intel MKL");
#endif
        return use_mkl;
      }
    }

    bool mayiuse_mkl() {
      static const bool mayiuse = mayiuse_mkl_init();
      return mayiuse;
    }

    std::string gemm_backend_to_str(GemmBackend gemm_backend) {
      switch (gemm_backend) {
      case GemmBackend::MKL:
        return "MKL";
      case GemmBackend::DNNL:
        return "DNNL";
      case GemmBackend::ACCELERATE:
        return "Accelerate";
      case GemmBackend::OPENBLAS:
        return "OpenBLAS";
      case GemmBackend::RUY:
        return "Ruy";
      default:
        return "none";
      }
    }

    GemmBackend get_gemm_backend(ComputeType compute_type) {
      const bool is_int8 = (compute_type == ComputeType::INT8
                            || compute_type == ComputeType::INT8_FLOAT32
                            || compute_type == ComputeType::INT8_FLOAT16
                            || compute_type == ComputeType::INT8_BFLOAT16);

#ifdef CT2_WITH_MKL
      if (mayiuse_mkl()
          && (compute_type == ComputeType::FLOAT32
              || compute_type == ComputeType::INT16
              || is_int8)) {
        return GemmBackend::MKL;
      }
#endif

#ifdef CT2_WITH_DNNL
      if (compute_type == ComputeType::FLOAT32 || is_int8) {
        return GemmBackend::DNNL;
      }
#endif

#ifdef CT2_WITH_ACCELERATE
      if (compute_type == ComputeType::FLOAT32) {
        return GemmBackend::ACCELERATE;
      }
#endif

#ifdef CT2_WITH_OPENBLAS
      if (compute_type == ComputeType::FLOAT32) {
        return GemmBackend::OPENBLAS;
      }
#endif

#ifdef CT2_WITH_RUY
      if (is_int8 || compute_type == ComputeType::FLOAT32) {
        return GemmBackend::RUY;
      }
#endif

      return GemmBackend::NONE;
    }

    bool has_gemm_backend(ComputeType compute_type) {
      return get_gemm_backend(compute_type) != GemmBackend::NONE;
    }

    bool prefer_u8s8s32_gemm() {
      const auto gemm_s8_backend = get_gemm_backend(ComputeType::INT8);
      return gemm_s8_backend == cpu::GemmBackend::MKL || gemm_s8_backend == cpu::GemmBackend::DNNL;
    }

    bool pack_gemm_weights(ComputeType compute_type) {
      static const bool should_pack = read_bool_from_env("CT2_USE_EXPERIMENTAL_PACKED_GEMM");
      return should_pack && get_gemm_backend(compute_type) == GemmBackend::MKL;
    }

#ifdef CT2_WITH_RUY
    ruy::Context *get_ruy_context() {
      static thread_local ruy::Context context;
      return &context;
    }
#endif
  }
}
