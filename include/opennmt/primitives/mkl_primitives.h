#pragma once

#include <mkl.h>

#include "generic_primitives.h"

namespace opennmt {
  namespace primitives {

    template<>
    void copy(const float* x, float* y, size_t size);
    template<>
    void add(float a, float* y, size_t size);
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
    void cos(const float* x, float* y, size_t size);
    template<>
    void sin(const float* x, float* y, size_t size);
    template<>
    void tanh(const float* x, float* y, size_t size);
    template<>
    void gemm(const float* a, const float* b,
              bool transpose_a, bool transpose_b,
              size_t m, size_t n, size_t k,
              float alpha, float beta,
              float* c);
    template<>
    void gemm_batch(const float* a, const float* b,
                    bool transpose_a, bool transpose_b,
                    size_t batch_size,
                    size_t m, size_t n, size_t k,
                    float alpha, float beta,
                    float* c);

    template <typename IndexType>
    void transpose_2d(const float* a, const IndexType* dims, float* b) {
      auto rows = dims[0];
      auto cols = dims[1];
      mkl_somatcopy('R', 'T', rows, cols, 1.0, a, cols, b, rows);
    }

  }
}
