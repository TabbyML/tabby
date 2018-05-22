#pragma once

#include <algorithm>

#include <mkl.h>

namespace onmt {
  namespace compute {

    template <typename T>
    void fill(T* x, T a, size_t size) {
      std::fill_n(x, size, a);
    }

    template <typename T>
    void copy(const T* x, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i)
        y[i] = x[i];
    }

    template <typename T>
    T sum(const T* array, size_t size) {
      return std::accumulate(array, array + size, static_cast<T>(0));
    }

    template <typename T>
    T mean(const T* array, size_t size) {
      return sum(array, size) / size;
    }

    template <typename T>
    size_t max_element(const T* array, size_t size) {
      return std::max_element(array, array + size) - array;
    }

    template <typename T>
    T max(const T* array, size_t size) {
      return array[max_element(array, size)];
    }

    template <typename T>
    void add(T a, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i)
        y[i] += a;
    }

    template <typename T>
    void add(const T* x, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i)
        y[i] += x[i];
    }

    template <typename T>
    void sub(T a, T* y, size_t size) {
      T a_rev = -a;
      add(a_rev, y, size);
    }

    template <typename T>
    void mul(T a, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i)
        y[i] *= a;
    }

    template <typename T>
    void mul(const T* x, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i)
        y[i] *= x[i];
    }

    // Functions without generic implementation.
    template <typename T>
    void pow(const T* x, T* y, T power, size_t size);
    template <typename T>
    void exp(const T* x, T* y, size_t size);
    template <typename In, typename Out>
    void gemm(const In* a, const In* b,
              bool transpose_a, bool transpose_b,
              size_t m, size_t n, size_t k,
              In alpha, Out beta,
              Out* c);

    template <typename In, typename Out>
    void gemm_batch(const In* a, const In* b,
                    bool transpose_a, bool transpose_b,
                    size_t batch_size,
                    size_t m, size_t n, size_t k,
                    In alpha, Out beta,
                    Out* c) {
      for (size_t i = 0; i < batch_size; ++i) {
        const In* a_i = a + (i * m * k);
        const In* b_i = b + (i * k * n);
        Out* c_i = c + (i * m * n);

        gemm(a_i, b_i, transpose_a, transpose_b, m, n, k, alpha, beta, c_i);
      }
    }


    // Specialization for CPU float.
    template<>
    void copy(const float* x, float* y, size_t size) {
      cblas_scopy(size, x, 1, y, 1);
    }
    template<>
    void add(float a, float* y, size_t size) {
      cblas_saxpy(size, 1.0, &a, 0, y, 1);
    }
    template<>
    void add(const float* x, float* y, size_t size) {
      cblas_saxpy(size, 1.0 /* a */, x, 1 /* incx */, y, 1 /* incy */);
    }
    template<>
    void mul(float a, float* y, size_t size) {
      cblas_sscal(size, a, y, 1);
    }
    template<>
    void mul(const float* x, float* y, size_t size) {
      vsMul(size, y, x, y);
    }
    template<>
    void pow(const float* x, float *y, float power, size_t size) {
      vsPowx(size, x, power, y);
    }
    template<>
    void exp(const float* x, float*y, size_t size) {
      vsExp(size, x, y);
    }

    template<>
    void gemm(const float* a, const float* b,
              bool transpose_a, bool transpose_b,
              size_t m, size_t n, size_t k,
              float alpha, float beta,
              float* c) {
      MKL_INT lda = transpose_a ? m : k;
      MKL_INT ldb = transpose_b ? k : n;
      MKL_INT ldc = n;

      MKL_INT m_ = m;
      MKL_INT n_ = n;
      MKL_INT k_ = k;

      CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
      CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;

      cblas_sgemm(CblasRowMajor,
                  trans_a, trans_b,
                  m_, n_, k_,
                  alpha, a, lda,
                  b, ldb,
                  beta, c, ldc);
    }

    template<>
    void gemm_batch(const float* a, const float* b,
                    bool transpose_a, bool transpose_b,
                    size_t batch_size,
                    size_t m, size_t n, size_t k,
                    float alpha, float beta,
                    float* c) {
      MKL_INT lda = transpose_a ? m : k;
      MKL_INT ldb = transpose_b ? k : n;
      MKL_INT ldc = n;

      MKL_INT b_ = batch_size;
      MKL_INT m_ = m;
      MKL_INT n_ = n;
      MKL_INT k_ = k;

      CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
      CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;

      std::vector<const float*> a_array(batch_size);
      std::vector<const float*> b_array(batch_size);
      std::vector<float*> c_array(batch_size);
      for (MKL_INT i = 0; i < b_; ++i) {
        a_array[i] = a + (i * m_ * k_);
        b_array[i] = b + (i * k_ * n_);
        c_array[i] = c + (i * m_ * n_);
      }

      cblas_sgemm_batch(CblasRowMajor,
                        &trans_a, &trans_b,
                        &m_, &n_, &k_,
                        &alpha, a_array.data(), &lda,
                        b_array.data(), &ldb,
                        &beta, c_array.data(), &ldc,
                        1 /* group_count */, &b_);
    }

  }
}
