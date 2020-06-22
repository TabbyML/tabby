#include "cpu_info.h"

#include <string>

#ifdef _WIN32
#  include <intrin.h>
#  include <immintrin.h>
#else
#  include <cpuid.h>
#endif

namespace ctranslate2 {
  namespace cpu {

    static void get_cpuid(unsigned int eax_in, unsigned int* data) {
#ifdef _WIN32
      __cpuid(reinterpret_cast<int*>(data), static_cast<int>(eax_in));
#else
      __cpuid(eax_in, data[0], data[1], data[2], data[3]);
#endif
    }

    static void get_cpuid_ex(unsigned int eax_in, unsigned int ecx_in, unsigned int* data) {
#ifdef _WIN32
      __cpuidex(reinterpret_cast<int*>(data), static_cast<int>(eax_in), static_cast<int>(ecx_in));
#else
      __cpuid_count(eax_in, ecx_in, data[0], data[1], data[2], data[3]);
#endif
    }

    static uint64_t get_bv()
    {
#ifdef _WIN32
      return _xgetbv(0);
#else
      unsigned int eax, edx;
      __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
      return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
    }

    struct CPUInfo {
      std::string vendor;
      bool is_intel = false;
      bool sse41 = false;
      bool avx = false;
      bool avx2 = false;

      CPUInfo() {
        unsigned int data[4];
        const unsigned int& eax = data[0];
        const unsigned int& ebx = data[1];
        const unsigned int& ecx = data[2];
        const unsigned int& edx = data[3];

        get_cpuid(0, data);
        const unsigned int max_num = eax;
        vendor = (std::string(reinterpret_cast<const char*>(&ebx), 4)
                  + std::string(reinterpret_cast<const char*>(&edx), 4)
                  + std::string(reinterpret_cast<const char*>(&ecx), 4));
        is_intel = (vendor == "GenuineIntel");

        get_cpuid(1, data);
        sse41 = ecx & (1 << 19);
        const bool osxsave = ecx & (1 << 27);
        if (osxsave && (get_bv() & 6) == 6) {
          avx = ecx & (1 << 28);
        }

        if (max_num >= 7) {
          get_cpuid_ex(7, 0, data);
          avx2 = avx && (ebx & (1 << 5));
        }
      }

    };

    static CPUInfo cpu_info;

    bool cpu_is_intel() {
      return cpu_info.is_intel;
    }

    bool cpu_supports_sse41() {
      return cpu_info.sse41;
    }

    bool cpu_supports_avx() {
      return cpu_info.avx;
    }

    bool cpu_supports_avx2() {
      return cpu_info.avx2;
    }

  }
}
