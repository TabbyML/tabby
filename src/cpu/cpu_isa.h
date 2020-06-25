#pragma once

#include <string>

namespace ctranslate2 {
  namespace cpu {

    enum class CpuIsa {
      GENERIC,
      AVX,
      AVX2,
    };

    std::string isa_to_str(CpuIsa isa);

    // Returns the CPU ISA to dispatch to.
    CpuIsa get_cpu_isa();

  }
}

#define CPU_ISA_CASE(CPU_ISA, STMTS)            \
  case CPU_ISA: {                               \
    const cpu::CpuIsa ISA = CPU_ISA;            \
    STMTS;                                      \
    break;                                      \
  }

#define CPU_ISA_DEFAULT(STMTS)                          \
  default: {                                            \
    const cpu::CpuIsa ISA = cpu::CpuIsa::GENERIC;       \
    STMTS;                                              \
    break;                                              \
  }

#ifdef CT2_WITH_CPU_DISPATCH
#  define CPU_ISA_DISPATCH(STMTS)                             \
  switch (cpu::get_cpu_isa()) {                               \
    CPU_ISA_CASE(cpu::CpuIsa::AVX2, SINGLE_ARG(STMTS))        \
    CPU_ISA_CASE(cpu::CpuIsa::AVX, SINGLE_ARG(STMTS))         \
    CPU_ISA_DEFAULT(SINGLE_ARG(STMTS))                        \
  }
#else
#  define CPU_ISA_DISPATCH(STMTS)                             \
  switch (cpu::get_cpu_isa()) {                               \
    CPU_ISA_DEFAULT(SINGLE_ARG(STMTS))                        \
  }
#endif
