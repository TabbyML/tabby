#include "opennmt/primitives/simd_primitives.h"

#include <cassert>

#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

namespace opennmt {
  namespace primitives {

#ifdef __AVX512F__

    template<>
    void gemm(const int16_t* a, const int16_t* b,
              bool transpose_a, bool transpose_b,
              size_t m, size_t n, size_t k,
              int16_t alpha, int32_t beta,
              int32_t* c) {
      assert(transpose_a == false);
      assert(transpose_b == true);

      const __m512i * A = reinterpret_cast<const __m512i*>(a);
      const __m512i * B = reinterpret_cast<const __m512i*>(b);

      int num_A_rows = m;
      int num_B_rows = n;
      int width = k;

      assert(width % 32 == 0);

      int avx_width = width / 32;

      // loop over A rows - 6 at a time
      int i;
      for (i = 0; i < num_A_rows - 5; i += 6) {
        const __m512i * A1_row = A + (i+0) * avx_width;
        const __m512i * A2_row = A + (i+1) * avx_width;
        const __m512i * A3_row = A + (i+2) * avx_width;
        const __m512i * A4_row = A + (i+3) * avx_width;
        const __m512i * A5_row = A + (i+4) * avx_width;
        const __m512i * A6_row = A + (i+5) * avx_width;

        for (int j = 0; j < num_B_rows; j++) {
          const __m512i * B_row = B + j * avx_width;

          __m512i sum1 = _mm512_setzero_si512();
          __m512i sum2 = _mm512_setzero_si512();
          __m512i sum3 = _mm512_setzero_si512();
          __m512i sum4 = _mm512_setzero_si512();
          __m512i sum5 = _mm512_setzero_si512();
          __m512i sum6 = _mm512_setzero_si512();

          // This is just a simple dot product, unrolled four ways.
          for (int k = 0; k < avx_width; k++) {
            __m512i b = *(B_row + k);

            __m512i a1 = *(A1_row + k);
            __m512i a2 = *(A2_row + k);
            __m512i a3 = *(A3_row + k);
            __m512i a4 = *(A4_row + k);
            __m512i a5 = *(A5_row + k);
            __m512i a6 = *(A6_row + k);

            // multiply and add
            sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(b, a1));
            sum2 = _mm512_add_epi32(sum2, _mm512_madd_epi16(b, a2));
            sum3 = _mm512_add_epi32(sum3, _mm512_madd_epi16(b, a3));
            sum4 = _mm512_add_epi32(sum4, _mm512_madd_epi16(b, a4));
            sum5 = _mm512_add_epi32(sum5, _mm512_madd_epi16(b, a5));
            sum6 = _mm512_add_epi32(sum6, _mm512_madd_epi16(b, a6));
          }

          int32_t * C1 = c + (i+0)*num_B_rows + j;
          (*C1) = beta * *C1 + alpha * _mm512_reduce_add_epi32(sum1);

          int32_t * C2 = c + (i+1)*num_B_rows + j;
          (*C2) = beta * *C2 + alpha * _mm512_reduce_add_epi32(sum2);

          int32_t * C3 = c + (i+2)*num_B_rows + j;
          (*C3) = beta * *C3 + alpha * _mm512_reduce_add_epi32(sum3);

          int32_t * C4 = c + (i+3)*num_B_rows + j;
          (*C4) = beta * *C4 + alpha * _mm512_reduce_add_epi32(sum4);

          int32_t * C5 = c + (i+4)*num_B_rows + j;
          (*C5) = beta * *C5 + alpha * _mm512_reduce_add_epi32(sum5);

          int32_t * C6 = c + (i+5)*num_B_rows + j;
          (*C6) = beta * *C6 + alpha * _mm512_reduce_add_epi32(sum6);

        }
      }
      // finalize the last rows
      switch (num_A_rows - i)
      {
      case 5:
      {
        const __m512i * A1_row = A + (i+0) * avx_width;
        const __m512i * A2_row = A + (i+1) * avx_width;
        const __m512i * A3_row = A + (i+2) * avx_width;
        const __m512i * A4_row = A + (i+3) * avx_width;
        const __m512i * A5_row = A + (i+4) * avx_width;

        for (int j = 0; j < num_B_rows; j++) {
          const __m512i * B_row = B + j * avx_width;

          __m512i sum1 = _mm512_setzero_si512();
          __m512i sum2 = _mm512_setzero_si512();
          __m512i sum3 = _mm512_setzero_si512();
          __m512i sum4 = _mm512_setzero_si512();
          __m512i sum5 = _mm512_setzero_si512();

          // This is just a simple dot product, unrolled four ways.
          for (int k = 0; k < avx_width; k++) {
            __m512i b = *(B_row + k);
            __m512i a1 = *(A1_row + k);
            __m512i a2 = *(A2_row + k);
            __m512i a3 = *(A3_row + k);
            __m512i a4 = *(A4_row + k);
            __m512i a5 = *(A5_row + k);

            // multiply and add
            sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(b, a1));
            sum2 = _mm512_add_epi32(sum2, _mm512_madd_epi16(b, a2));
            sum3 = _mm512_add_epi32(sum3, _mm512_madd_epi16(b, a3));
            sum4 = _mm512_add_epi32(sum4, _mm512_madd_epi16(b, a4));
            sum5 = _mm512_add_epi32(sum5, _mm512_madd_epi16(b, a5));
          }

          int32_t * C1 = c + (i+0)*num_B_rows + j;
          (*C1) = beta * *C1 + alpha * _mm512_reduce_add_epi32(sum1);

          int32_t * C2 = c + (i+1)*num_B_rows + j;
          (*C2) = beta * *C2 + alpha * _mm512_reduce_add_epi32(sum2);

          int32_t * C3 = c + (i+2)*num_B_rows + j;
          (*C3) = beta * *C3 + alpha * _mm512_reduce_add_epi32(sum3);

          int32_t * C4 = c + (i+3)*num_B_rows + j;
          (*C4) = beta * *C4 + alpha * _mm512_reduce_add_epi32(sum4);

          int32_t * C5 = c + (i+4)*num_B_rows + j;
          (*C5) = beta * *C5 + alpha * _mm512_reduce_add_epi32(sum5);
        }
      }
      break;
      case 4:
      {
        const __m512i * A1_row = A + (i+0) * avx_width;
        const __m512i * A2_row = A + (i+1) * avx_width;
        const __m512i * A3_row = A + (i+2) * avx_width;
        const __m512i * A4_row = A + (i+3) * avx_width;

        for (int j = 0; j < num_B_rows; j++) {
          const __m512i * B_row = B + j * avx_width;

          __m512i sum1 = _mm512_setzero_si512();
          __m512i sum2 = _mm512_setzero_si512();
          __m512i sum3 = _mm512_setzero_si512();
          __m512i sum4 = _mm512_setzero_si512();

          // This is just a simple dot product, unrolled four ways.
          for (int k = 0; k < avx_width; k++) {
            __m512i b = *(B_row + k);
            __m512i a1 = *(A1_row + k);
            __m512i a2 = *(A2_row + k);
            __m512i a3 = *(A3_row + k);
            __m512i a4 = *(A4_row + k);

            // multiply and add
            sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(b, a1));
            sum2 = _mm512_add_epi32(sum2, _mm512_madd_epi16(b, a2));
            sum3 = _mm512_add_epi32(sum3, _mm512_madd_epi16(b, a3));
            sum4 = _mm512_add_epi32(sum4, _mm512_madd_epi16(b, a4));
          }

          int32_t * C1 = c + (i+0)*num_B_rows + j;
          (*C1) = beta * *C1 + alpha * _mm512_reduce_add_epi32(sum1);

          int32_t * C2 = c + (i+1)*num_B_rows + j;
          (*C2) = beta * *C2 + alpha * _mm512_reduce_add_epi32(sum2);

          int32_t * C3 = c + (i+2)*num_B_rows + j;
          (*C3) = beta * *C3 + alpha * _mm512_reduce_add_epi32(sum3);

          int32_t * C4 = c + (i+3)*num_B_rows + j;
          (*C4) = beta * *C4 + alpha * _mm512_reduce_add_epi32(sum4);
        }
      }
      break;
      case 3:
      {
        const __m512i * A1_row = A + (i+0) * avx_width;
        const __m512i * A2_row = A + (i+1) * avx_width;
        const __m512i * A3_row = A + (i+2) * avx_width;

        for (int j = 0; j < num_B_rows; j++) {
          const __m512i * B_row = B + j * avx_width;

          __m512i sum1 = _mm512_setzero_si512();
          __m512i sum2 = _mm512_setzero_si512();
          __m512i sum3 = _mm512_setzero_si512();

          // This is just a simple dot product, unrolled four ways.
          for (int k = 0; k < avx_width; k++) {
            __m512i b = *(B_row + k);
            __m512i a1 = *(A1_row + k);
            __m512i a2 = *(A2_row + k);
            __m512i a3 = *(A3_row + k);

            // multiply and add
            sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(b, a1));
            sum2 = _mm512_add_epi32(sum2, _mm512_madd_epi16(b, a2));
            sum3 = _mm512_add_epi32(sum3, _mm512_madd_epi16(b, a3));
          }

          int32_t * C1 = c + (i+0)*num_B_rows + j;
          (*C1) = beta * *C1 + alpha * _mm512_reduce_add_epi32(sum1);

          int32_t * C2 = c + (i+1)*num_B_rows + j;
          (*C2) = beta * *C2 + alpha * _mm512_reduce_add_epi32(sum2);

          int32_t * C3 = c + (i+2)*num_B_rows + j;
          (*C3) = beta * *C3 + alpha * _mm512_reduce_add_epi32(sum3);
        }
      }
      break;
      case 2:
      {
        const __m512i * A1_row = A + (i+0) * avx_width;
        const __m512i * A2_row = A + (i+1) * avx_width;

        for (int j = 0; j < num_B_rows; j++) {
          const __m512i * B_row = B + j * avx_width;

          __m512i sum1 = _mm512_setzero_si512();
          __m512i sum2 = _mm512_setzero_si512();

          // This is just a simple dot product, unrolled four ways.
          for (int k = 0; k < avx_width; k++) {
            __m512i b = *(B_row + k);
            __m512i a1 = *(A1_row + k);
            __m512i a2 = *(A2_row + k);

            // multiply and add
            sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(b, a1));
            sum2 = _mm512_add_epi32(sum2, _mm512_madd_epi16(b, a2));
          }

          int32_t * C1 = c + (i+0)*num_B_rows + j;
          (*C1) = beta * *C1 + alpha * _mm512_reduce_add_epi32(sum1);

          int32_t * C2 = c + (i+1)*num_B_rows + j;
          (*C2) = beta * *C2 + alpha * _mm512_reduce_add_epi32(sum2);
        }
      }
      break;
      case 1:
      {
        const __m512i * A1_row = A + (i+0) * avx_width;

        for (int j = 0; j < num_B_rows; j++) {
          const __m512i * B_row = B + j * avx_width;

          __m512i sum1 = _mm512_setzero_si512();

          // This is just a simple dot product, unrolled four ways.
          for (int k = 0; k < avx_width; k++) {
            __m512i b = *(B_row + k);
            __m512i a1 = *(A1_row + k);

            // multiply and add
            sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(b, a1));
          }

          int32_t * C1 = c + (i+0)*num_B_rows + j;
          (*C1) = beta * *C1 + alpha * _mm512_reduce_add_epi32(sum1);
        }
      }
      break;
      }
    }

#elif __AVX2__

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
