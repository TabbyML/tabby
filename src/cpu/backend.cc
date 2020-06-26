#include "backend.h"

#ifdef WITH_MKL
#  include <mkl.h>
#endif

#include "ctranslate2/utils.h"
#include "cpu_info.h"

namespace ctranslate2 {
  namespace cpu {

#ifdef WITH_MKL
    static inline bool mkl_has_fast_int_gemm() {
#  if __INTEL_MKL__ > 2019 || (__INTEL_MKL__ == 2019 && __INTEL_MKL_UPDATE__ >= 5)
      // Intel MKL 2019.5 added optimized integers GEMM for SSE4.2 and AVX (in addition to
      // the existing AVX2 and AVX512), so it is virtually optimized for all target platforms.
      return true;
#  else
      return mkl_cbwr_get_auto_branch() >= MKL_CBWR_AVX2;
#  endif
    }
#endif

    static bool mayiuse_mkl_init() {
      const std::string use_mkl_env = read_string_from_env("CT2_USE_MKL");
      if (use_mkl_env.empty()) {
#ifdef WITH_MKL
        return cpu_is_intel();
#else
        return false;
#endif
      } else {
        const bool use_mkl = string_to_bool(use_mkl_env);
#ifndef WITH_MKL
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
      default:
        return "NONE";
      }
    }

    GemmBackend get_gemm_backend(ComputeType compute_type) {
#ifdef WITH_DNNL
      if (!mayiuse_mkl()) {
        if (compute_type != ComputeType::INT16)
          return GemmBackend::DNNL;
        else
          return GemmBackend::NONE;
      }
#endif
#ifdef WITH_MKL
      if (compute_type == ComputeType::FLOAT || mkl_has_fast_int_gemm()) {
        return GemmBackend::MKL;
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

    bool should_pack_gemm_weights() {
      static const bool should_pack = read_bool_from_env("CT2_USE_EXPERIMENTAL_PACKED_GEMM");
      return should_pack;
    }

  }
}
