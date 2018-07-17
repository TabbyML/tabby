#pragma once

#include "cpu_generic.h"

namespace opennmt {
  namespace primitives {

    // MKL specializations.

    template<>
    void copy(const float* x, float* y, size_t size);
    template<>
    void add(const float* x, float* y, size_t size);
    template<>
    void add(const float* a, const float* b, float* c, size_t size);
    template<>
    void sub(const float* a, const float* b, float* c, size_t size);
    template<>
    void mul(float a, float* y, size_t size);
    template<>
    void mul(const float* x, float* y, size_t size);
    template<>
    void mul(const float* a, const float* b, float* c, size_t size);
    template<>
    void inv(const float* x, float* y, size_t size);
    template<>
    void pow(const float* x, float *y, float power, size_t size);
    template<>
    void exp(const float* x, float* y, size_t size);
    template<>
    void tanh(const float* x, float* y, size_t size);
    template<>
    void transpose_2d(const float* a, const size_t* dims, float* b);
    template<>
    void gemm(const float* a, const float* b,
              bool transpose_a, bool transpose_b,
              size_t m, size_t n, size_t k,
              float alpha, float beta,
              float* c);
    template<>
    void gemm(const int16_t* a, const int16_t* b,
              bool transpose_a, bool transpose_b,
              size_t m, size_t n, size_t k,
              int16_t alpha, int32_t beta,
              int32_t* c);
    template<>
    void gemm_batch(const float* a, const float* b,
                    bool transpose_a, bool transpose_b,
                    size_t batch_size,
                    size_t m, size_t n, size_t k,
                    float alpha, float beta,
                    float* c);

  }
}
