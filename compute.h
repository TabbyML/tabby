#pragma once

#include <algorithm>

#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

#include <mkl.h>

namespace onmt {
  namespace compute {

    template <typename T>
    void fill(T* x, T a, size_t size) {
      std::fill_n(x, size, a);
    }

    template <typename T>
    void copy(const T* x, T* y, size_t size) {
      std::copy_n(x, size, y);
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

    template <typename T>
    void mul(const T* a, const T* b, T* c, size_t size) {
      for (size_t i = 0; i < size; ++i)
        c[i] = a[i] * b[i];
    }

    template <typename T>
    void inv(const T* x, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i)
        y[i] = static_cast<T>(1) / x[i];
    }

    template <typename In, typename Out>
    void quantize(const In* x, Out* y, size_t size, In scale, In shift) {
      for (size_t i = 0; i < size; ++i)
        y[i] = static_cast<Out>(x[i] * scale + shift);
    }

    template <typename In, typename Out>
    void unquantize(const In* x, Out* y, size_t size, Out scale, Out shift) {
      for (size_t i = 0; i < size; ++i)
        y[i] = (static_cast<Out>(x[i]) - shift) / scale;
    }

    template <typename T>
    void relu(T* x, size_t size) {
      for (size_t i = 0; i < size; ++i) {
        if (x[i] < static_cast<T>(0))
          x[i] = static_cast<T>(0);
      }
    }

    template <typename T>
    void relu(const T* x, T* y, size_t size) {
      for (size_t i = 0; i < size; ++i) {
        y[i] = x[i] > 0 ? x[i] : static_cast<T>(0);
      }
    }


    // Functions without generic implementation.
    template <typename T>
    void pow(const T* x, T* y, T power, size_t size);
    template <typename T>
    void exp(const T* x, T* y, size_t size);
    template <typename T>
    void cos(const T* x, T* y, size_t size);
    template <typename T>
    void sin(const T* x, T* y, size_t size);
    template <typename T>
    void tanh(const T* x, T* y, size_t size);
    template <typename In, typename Out>
    void gemm(const In* a, const In* b,
              bool transpose_a, bool transpose_b,
              size_t m, size_t n, size_t k,
              In alpha, Out beta,
              Out* c);
    template <typename T>
    void transpose_2d_inplace(T* a, size_t rows, size_t cols);
    template <typename T>
    void transpose_2d(const T* a, size_t rows, size_t cols, T* b);

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


    // Intel MKL specialization for standard float.
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
    void mul(const float* a, const float* b, float* c, size_t size) {
      vsMul(size, a, b, c);
    }
    template<>
    void inv(const float* x, float* y, size_t size) {
      vsInv(size, x, y);
    }
    template<>
    void pow(const float* x, float *y, float power, size_t size) {
      vsPowx(size, x, power, y);
    }
    template<>
    void exp(const float* x, float* y, size_t size) {
      vsExp(size, x, y);
    }
    template<>
    void cos(const float* x, float* y, size_t size) {
      vsCos(size, x, y);
    }
    template<>
    void sin(const float* x, float* y, size_t size) {
      vsSin(size, x, y);
    }
    template<>
    void tanh(const float* x, float* y, size_t size) {
      vsTanh(size, x, y);
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

    template<>
    void transpose_2d_inplace(float* a, size_t rows, size_t cols) {
      mkl_simatcopy('R', 'T', rows, cols, 1.0, a, cols, rows);
    }
    template<>
    void transpose_2d(const float* a, size_t rows, size_t cols, float* b) {
      mkl_somatcopy('R', 'T', rows, cols, 1.0, a, cols, b, rows);
    }

#ifdef __AVX2__
    /* horizontal i32 add on __m256 - returns m128i register */
    static inline __m128i _mm256i_sum8 (__m256i a)
    {
      // add 2*8
      a  = _mm256_hadd_epi32(a, a);
      // add again - low and high are summed now
      a  = _mm256_hadd_epi32(a, a);
      // add low and high part
      return _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    }

    template<>
    void gemm(const int16_t* a, const int16_t* b,
              bool transpose_a, bool transpose_b,
              size_t m, size_t n, size_t k,
              int16_t alpha, int32_t beta,
              int32_t* c) {
      assert(transpose_a == false);
      assert(transpose_b == true);

      const __m256i * A = reinterpret_cast<const __m256i*>(a);
      const __m256i * B = reinterpret_cast<const __m256i*>(b);

      int num_A_rows = m;
      int num_B_rows = n;
      int width = k;

      assert(width % 16 == 0);

      int avx_width = width / 16;

      // loop over A rows - 4 at a time
      int i;
      for (i = 0; i < num_A_rows - 3; i += 4)
      {
        const __m256i * A1_row = A + (i+0) * avx_width;
        const __m256i * A2_row = A + (i+1) * avx_width;
        const __m256i * A3_row = A + (i+2) * avx_width;
        const __m256i * A4_row = A + (i+3) * avx_width;

        for (int j = 0; j < num_B_rows; j++)
        {
          const __m256i * B_row = B + j * avx_width;

          __m256i sum1 = _mm256_setzero_si256();
          __m256i sum2 = _mm256_setzero_si256();
          __m256i sum3 = _mm256_setzero_si256();
          __m256i sum4 = _mm256_setzero_si256();

          // This is just a simple dot product, unrolled four ways.
          for (int k = 0; k < avx_width; k++)
          {
            __m256i b = *(B_row + k);

            __m256i a1 = *(A1_row + k);
            __m256i a2 = *(A2_row + k);
            __m256i a3 = *(A3_row + k);
            __m256i a4 = *(A4_row + k);

            // multiply and add
            sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(b, a1));
            sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(b, a2));
            sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(b, a3));
            sum4 = _mm256_add_epi32(sum4, _mm256_madd_epi16(b, a4));
          }

          // horizontal add
          __m128i sum1_128 = _mm256i_sum8(sum1);
          __m128i sum2_128 = _mm256i_sum8(sum2);
          __m128i sum3_128 = _mm256i_sum8(sum3);
          __m128i sum4_128 = _mm256i_sum8(sum4);

          int32_t * C1 = c + (i+0)*num_B_rows + j;
          int32_t * C2 = c + (i+1)*num_B_rows + j;
          int32_t * C3 = c + (i+2)*num_B_rows + j;
          int32_t * C4 = c + (i+3)*num_B_rows + j;

          *C1 = beta * *C1 + alpha * _mm_cvtsi128_si32(sum1_128);
          *C2 = beta * *C2 + alpha * _mm_cvtsi128_si32(sum2_128);
          *C3 = beta * *C3 + alpha * _mm_cvtsi128_si32(sum3_128);
          *C4 = beta * *C4 + alpha * _mm_cvtsi128_si32(sum4_128);
        }
      }
      // finalize the last rows
      switch (num_A_rows - i)
      {
      case 3:
      {
        const __m256i * A1_row = A + (i+0) * avx_width;
        const __m256i * A2_row = A + (i+1) * avx_width;
        const __m256i * A3_row = A + (i+2) * avx_width;
        for (int j = 0; j < num_B_rows; j++)
        {
          const __m256i * B_row = B + j * avx_width;
          __m256i sum1 = _mm256_setzero_si256();
          __m256i sum2 = _mm256_setzero_si256();
          __m256i sum3 = _mm256_setzero_si256();
          for (int k = 0; k < avx_width; k++)
          {
            __m256i b = *(B_row + k);
            __m256i a1 = *(A1_row + k);
            __m256i a2 = *(A2_row + k);
            __m256i a3 = *(A3_row + k);
            sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(b, a1));
            sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(b, a2));
            sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(b, a3));
          }
          __m128i sum1_128 = _mm256i_sum8(sum1);
          __m128i sum2_128 = _mm256i_sum8(sum2);
          __m128i sum3_128 = _mm256i_sum8(sum3);
          int32_t * C1 = c + (i+0)*num_B_rows + j;
          int32_t * C2 = c + (i+1)*num_B_rows + j;
          int32_t * C3 = c + (i+2)*num_B_rows + j;
          *C1 = beta * *C1 + alpha * _mm_cvtsi128_si32(sum1_128);
          *C2 = beta * *C2 + alpha * _mm_cvtsi128_si32(sum2_128);
          *C3 = beta * *C3 + alpha * _mm_cvtsi128_si32(sum3_128);
        }
      }
      break;
      case 2:
      {
        const __m256i * A1_row = A + (i+0) * avx_width;
        const __m256i * A2_row = A + (i+1) * avx_width;
        for (int j = 0; j < num_B_rows; j++)
        {
          const __m256i * B_row = B + j * avx_width;
          __m256i sum1 = _mm256_setzero_si256();
          __m256i sum2 = _mm256_setzero_si256();
          for (int k = 0; k < avx_width; k++)
          {
            __m256i b = *(B_row + k);
            __m256i a1 = *(A1_row + k);
            __m256i a2 = *(A2_row + k);
            sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(b, a1));
            sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(b, a2));
          }
          __m128i sum1_128 = _mm256i_sum8(sum1);
          __m128i sum2_128 = _mm256i_sum8(sum2);
          int32_t * C1 = c + (i+0)*num_B_rows + j;
          int32_t * C2 = c + (i+1)*num_B_rows + j;
          *C1 = beta * *C1 + alpha * _mm_cvtsi128_si32(sum1_128);
          *C2 = beta * *C2 + alpha * _mm_cvtsi128_si32(sum2_128);
        }
      }
      break;
      case 1:
      {
        const __m256i * A1_row = A + (i+0) * avx_width;
        for (int j = 0; j < num_B_rows; j++)
        {
          const __m256i * B_row = B + j * avx_width;
          __m256i sum1 = _mm256_setzero_si256();
          for (int k = 0; k < avx_width; k++)
          {
            __m256i b = *(B_row + k);
            __m256i a1 = *(A1_row + k);
            sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(b, a1));
          }
          __m128i sum1_128 = _mm256i_sum8(sum1);
          int32_t * C1 = c + (i+0)*num_B_rows + j;
          *C1 = beta * *C1 + alpha * _mm_cvtsi128_si32(sum1_128);
        }
      }
      break;
      default:
        break;
      }
    }
#endif

  }
}
