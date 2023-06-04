#pragma once

namespace ctranslate2 {
  namespace cpu {

    // Functions returning some info about the current CPU.

    const char* cpu_vendor();
#if defined(CT2_X86_BUILD)
    bool cpu_is_genuine_intel();
    bool cpu_supports_sse41();
    bool cpu_supports_avx();
    bool cpu_supports_avx2();
    bool cpu_supports_avx512();
#elif defined(CT2_ARM64_BUILD)
    bool cpu_supports_neon();
#endif

  }
}
