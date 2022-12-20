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
    template <CpuIsa ISA>
    void gelu(const float* x, float* y, dim_t size);
    template <CpuIsa ISA>
    void swish(const float* x, float* y, dim_t size);

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

    template <CpuIsa ISA, typename T>
    T reduce_sum(const T* x, dim_t size);
    template <CpuIsa ISA, typename T>
    T reduce_max(const T* x, dim_t size);
    template <CpuIsa ISA, typename T>
    T reduce_amax(const T* x, dim_t size);

    template <CpuIsa ISA>
    float reduce_logsumexp(const float* x, dim_t size);

    template <CpuIsa ISA>
    void softmax(const float* input,
                 const int32_t* lengths,
                 float* output,
                 dim_t batch_size,
                 dim_t depth,
                 bool log,
                 float epsilon);

    template <CpuIsa ISA>
    void layer_norm(const float* input,
                    const float* gamma,
                    const float* beta,
                    float* output,
                    dim_t batch_size,
                    dim_t depth,
                    float epsilon);

    template <CpuIsa ISA>
    void rms_norm(const float* input,
                  const float* gamma,
                  float* output,
                  dim_t batch_size,
                  dim_t depth,
                  float epsilon);

    template <CpuIsa ISA>
    void quantize_s8(const float* x,
                     int8_t* y,
                     float* scales,
                     dim_t batch_size,
                     dim_t depth,
                     bool shift_to_uint8,
                     bool round_before_cast);

    struct identity {
      template <typename T>
      constexpr T&& operator()(T&& v) const noexcept {
        return std::forward<T>(v);
      }
    };

  }
}
