#pragma once

#include "ctranslate2/types.h"
#include "cpu_isa.h"

namespace ctranslate2 {
  namespace cpu {

    template <CpuIsa ISA, typename T>
    void rcp(const T* x, T* y, dim_t size);

    template <CpuIsa ISA>
    void exp(const float* x, float* y, dim_t size);
    template <CpuIsa ISA>
    void log(const float* x, float* y, dim_t size);
    template <CpuIsa ISA>
    void sin(const float* x, float* y, dim_t size);
    template <CpuIsa ISA>
    void cos(const float* x, float* y, dim_t size);

    template <CpuIsa ISA, typename T>
    void add(T a, const T* x, T* y, dim_t size);
    template <CpuIsa ISA, typename T>
    void add(const T* a, const T* b, T* c, dim_t size);

    template <CpuIsa ISA, typename T>
    void sub(const T* a, const T* b, T* c, dim_t size);

    template <CpuIsa ISA, typename T>
    void mul(T a, const T* x, T* y, dim_t size);
    template <CpuIsa ISA, typename T>
    void mul(const T* a, const T* b, T* c, dim_t size);

    template <CpuIsa ISA, typename T>
    void max(T a, const T* x, T* y, dim_t size);
    template <CpuIsa ISA, typename T>
    void max(const T* a, const T* b, T* c, dim_t size);

    template <CpuIsa ISA, typename T>
    void min(T a, const T* x, T* y, dim_t size);
    template <CpuIsa ISA, typename T>
    void min(const T* a, const T* b, T* c, dim_t size);

  }
}
