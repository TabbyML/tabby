#pragma once

namespace ctranslate2 {
  namespace cpu {

    // Functions returning some info about the current CPU.

    bool cpu_is_intel();
    bool cpu_supports_sse41();
    bool cpu_supports_avx();
    bool cpu_supports_avx2();

  }
}
