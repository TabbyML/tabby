#include "cpu_isa.h"

#include "ctranslate2/utils.h"
#include "cpu_info.h"

namespace ctranslate2 {
  namespace cpu {

    static CpuIsa try_isa(const std::string& name, CpuIsa cpu_isa, bool supported) {
#ifdef CT2_WITH_CPU_DISPATCH
      if (!supported) {
        throw std::invalid_argument("The CPU does not support " + name);
      } else {
        return cpu_isa;
      }
#else
      (void)cpu_isa;
      (void)supported;
      throw std::invalid_argument("This CTranslate2 binary was not compiled with "
                                  + name + " support");
#endif
    }

    std::string isa_to_str(CpuIsa isa) {
      switch (isa) {
      case CpuIsa::AVX:
        return "AVX";
      case CpuIsa::AVX2:
        return "AVX2";
      default:
        return "GENERIC";
      }
    }

    static CpuIsa init_isa() {
      const std::string env_isa = read_string_from_env("CT2_FORCE_CPU_ISA");
      if (!env_isa.empty()) {
        if (env_isa == "AVX2") {
          return try_isa(env_isa, CpuIsa::AVX2, cpu_supports_avx2());
        } else if (env_isa == "AVX") {
          return try_isa(env_isa, CpuIsa::AVX, cpu_supports_avx());
        } else if (env_isa == "GENERIC") {
          return CpuIsa::GENERIC;
        } else {
          throw std::invalid_argument("Invalid CPU ISA: " + env_isa);
        }
      }

#ifdef CT2_WITH_CPU_DISPATCH
      if (cpu_supports_avx2()) {
        return CpuIsa::AVX2;
      } else if (cpu_supports_avx()) {
        return CpuIsa::AVX;
      }
#endif

      return CpuIsa::GENERIC;
    }

    CpuIsa get_cpu_isa() {
      static CpuIsa cpu_isa = init_isa();
      return cpu_isa;
    }

  }
}
