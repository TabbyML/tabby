#include "cpu_info.h"

#if defined(CT2_X86_BUILD)

#include <cpuinfo_x86.h>

namespace ctranslate2 {
  namespace cpu {

    static const cpu_features::X86Info info = cpu_features::GetX86Info();

    const std::string& cpu_vendor() {
      static const std::string vendor = info.vendor;
      return vendor;
    }

    bool cpu_supports_sse41() {
      return info.features.sse4_1;
    }

    bool cpu_supports_avx() {
      return info.features.avx;
    }

    bool cpu_supports_avx2() {
      return info.features.avx2;
    }

  }
}

#elif defined(CT2_ARM64_BUILD)

namespace ctranslate2 {
  namespace cpu {

    const std::string& cpu_vendor() {
      static const std::string vendor = "ARM";
      return vendor;
    }

    bool cpu_supports_neon() {
      return true;
    }

  }
}

#endif
