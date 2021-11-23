#pragma once

#include <immintrin.h>
#include <avx_mathfun.h>

#include "vec.h"

#if defined(__GNUC__)
#  define __ct2_align32__ __attribute__((aligned(32)))
#elif defined(_WIN32)
#  define __ct2_align32__ __declspec(align(32))
#else
#  define __ct2_align32__
#endif

namespace ctranslate2 {
  namespace cpu {

    template<>
    struct Vec<float, TARGET_ISA> {

      using value_type = __m256;
      static constexpr dim_t width = 8;

      static inline value_type load(float value) {
        return _mm256_set1_ps(value);
      }

      static inline value_type load(const float* ptr) {
        return _mm256_loadu_ps(ptr);
      }

      static inline value_type load(const float* ptr, dim_t count, float default_value = 0) {
        if (count == width) {
          return _mm256_loadu_ps(ptr);
        } else {
          __ct2_align32__ float tmp_values[width];
          std::fill(tmp_values, tmp_values + width, default_value);
          std::copy(ptr, ptr + count, tmp_values);
          return _mm256_loadu_ps(tmp_values);
        }
      }

      static inline void store(value_type value, float* ptr) {
        _mm256_storeu_ps(ptr, value);
      }

      static inline void store(value_type value, float* ptr, dim_t count) {
        if (count == width) {
          _mm256_storeu_ps(ptr, value);
        } else {
          __ct2_align32__ float tmp_values[width];
          _mm256_storeu_ps(tmp_values, value);
          std::copy(tmp_values, tmp_values + count, ptr);
        }
      }

      static inline value_type abs(value_type a) {
        auto mask = _mm256_set1_ps(-0.f);
        return _mm256_andnot_ps(mask, a);
      }

      static inline value_type rcp(value_type a) {
        return _mm256_rcp_ps(a);
      }

      static inline value_type exp(value_type a) {
        return exp256_ps(a);
      }

      static inline value_type log(value_type a) {
        return log256_ps(a);
      }

      static inline value_type sin(value_type a) {
        return sin256_ps(a);
      }

      static inline value_type cos(value_type a) {
        return cos256_ps(a);
      }

      static inline value_type max(value_type a, value_type b) {
        return _mm256_max_ps(a, b);
      }

      static inline value_type min(value_type a, value_type b) {
        return _mm256_min_ps(a, b);
      }

      static inline value_type add(value_type a, value_type b) {
        return _mm256_add_ps(a, b);
      }

      static inline value_type sub(value_type a, value_type b) {
        return _mm256_sub_ps(a, b);
      }

      static inline value_type mul(value_type a, value_type b) {
        return _mm256_mul_ps(a, b);
      }

    };

  }
}
