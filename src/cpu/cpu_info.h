#pragma once

#include <string>

namespace ctranslate2 {
  namespace cpu {

    // Functions returning some info about the current CPU.

    const std::string& cpu_vendor();
    bool cpu_supports_sse41();
    bool cpu_supports_avx();
    bool cpu_supports_avx2();
    bool cpu_supports_neon();

  }
}
