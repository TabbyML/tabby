#pragma once

#include <string>

namespace ctranslate2 {
  namespace cpu {

    // Functions returning some info about the current CPU.

    const std::string& cpu_vendor();
#if defined(CT2_X86_BUILD)
    bool cpu_supports_sse41();
    bool cpu_supports_avx();
    bool cpu_supports_avx2();
#elif defined(CT2_ARM64_BUILD)
    bool cpu_supports_neon();
#endif

  }
}
