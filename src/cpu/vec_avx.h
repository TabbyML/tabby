#pragma once

// __FMA__ is not defined in MSVC, however it is implied with AVX2.
#if defined(_MSC_VER) && defined(__AVX2__)
#  ifndef __FMA__
#  define __FMA__
#  endif
#endif

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

    template <typename Func>
    float reduce_m256(__m256 v, const Func& func) {
      // Code adapted from https://github.com/pytorch/pytorch/blob/v1.12.0/aten/src/ATen/cpu/vec/functional_base.h#L41-L54

      // 128-bit shuffle
      auto v1 = _mm256_permute2f128_ps(v, v, 0x1);
      v = func(v, v1);
      // 64-bit shuffle
      v1 = _mm256_shuffle_ps(v, v, 0x4E);
      v = func(v, v1);
      // 32-bit shuffle
      v1 = _mm256_shuffle_ps(v, v, 0xB1);
      v = func(v, v1);
      return _mm256_cvtss_f32(v);
    }

    template<>
    struct Vec<float, TARGET_ISA> {

      using value_type = __m256;
      using mask_type = value_type;
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

      static inline value_type load_and_convert(const int32_t* ptr) {
        return _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
      }

      static inline value_type load_and_convert(const int32_t* ptr,
                                                dim_t count,
                                                int32_t default_value = 0) {
        if (count == width) {
          return load_and_convert(ptr);
        } else {
          __ct2_align32__ int32_t tmp_values[width];
          std::fill(tmp_values, tmp_values + width, default_value);
          std::copy(ptr, ptr + count, tmp_values);
          return load_and_convert(tmp_values);
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

      static inline value_type bit_and(value_type a, value_type b) {
        return _mm256_and_ps(a, b);
      }

      static inline value_type bit_xor(value_type a, value_type b) {
        return _mm256_xor_ps(a, b);
      }

      static inline value_type lt(value_type a, value_type b) {
        return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
      }

      static inline value_type select(value_type mask, value_type a, value_type b) {
        return _mm256_blendv_ps(b, a, mask);
      }

      static inline value_type abs(value_type a) {
        auto mask = _mm256_set1_ps(-0.f);
        return _mm256_andnot_ps(mask, a);
      }

      static inline value_type neg(value_type a) {
        return _mm256_xor_ps(a, _mm256_set1_ps(-0.f));
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

      static inline value_type tanh(value_type a) {
        return vec_tanh<TARGET_ISA>(a);
      }

      static inline value_type erf(value_type a) {
        return vec_erf<TARGET_ISA>(a);
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

      static inline value_type div(value_type a, value_type b) {
        return _mm256_div_ps(a, b);
      }

      static inline value_type mul_add(value_type a, value_type b, value_type c) {
#ifdef __FMA__
        return _mm256_fmadd_ps(a, b, c);
#else
        return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
      }

      static inline float reduce_add(value_type a) {
        return reduce_m256(a, add);
      }

      static inline float reduce_max(value_type a) {
        return reduce_m256(a, max);
      }

      static inline value_type round(value_type a) {
          return _mm256_round_ps(a, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
      }

      template<typename T>
      static void convert_and_store(value_type v, T* a, dim_t count) {
        auto i32 = _mm256_cvttps_epi32(v);
        int32_t  tmp[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(tmp), i32);
        std::copy(tmp, tmp + count, a);
      }
    };

  }
}
