#pragma once

#include <immintrin.h>
#include <avx512_mathfun.h>

#include "vec.h"

namespace ctranslate2 {
  namespace cpu {

    template<>
    struct Vec<float, TARGET_ISA> {

      using value_type = __m512;
      using mask_type = __mmask16;
      static constexpr dim_t width = 16;

      static inline mask_type get_length_mask(int length) {
        const int mask = (1 << length) - 1;
        return _mm512_int2mask(mask);
      }

      static inline value_type load(float value) {
        return _mm512_set1_ps(value);
      }

      static inline value_type load(const float* ptr) {
        return _mm512_loadu_ps(ptr);
      }

      static inline value_type load(const float* ptr, dim_t count, float default_value = 0) {
        value_type padding = load(default_value);
        mask_type mask = get_length_mask(count);
        return _mm512_mask_loadu_ps(padding, mask, ptr);
      }

      static inline value_type load_and_convert(const int32_t* ptr) {
        return _mm512_cvtepi32_ps(_mm512_loadu_si512(ptr));
      }

      static inline value_type load_and_convert(const int32_t* ptr,
                                                dim_t count,
                                                int32_t default_value = 0) {
        auto padding = _mm512_set1_epi32(default_value);
        mask_type mask = get_length_mask(count);
        return _mm512_cvtepi32_ps(_mm512_mask_loadu_epi32(padding, mask, ptr));
      }

      static inline void store(value_type value, float* ptr) {
        _mm512_storeu_ps(ptr, value);
      }

      static inline void store(value_type value, float* ptr, dim_t count) {
        mask_type mask = get_length_mask(count);
        _mm512_mask_storeu_ps(ptr, mask, value);
      }

      static inline value_type bit_and(value_type a, value_type b) {
        return _mm512_and_ps(a, b);
      }

      static inline value_type bit_xor(value_type a, value_type b) {
        return _mm512_xor_ps(a, b);
      }

      static inline mask_type lt(value_type a, value_type b) {
        return _mm512_cmp_ps_mask(a, b, _CMP_LT_OS);
      }

      static inline value_type select(mask_type mask, value_type a, value_type b) {
        return _mm512_mask_blend_ps(mask, b, a);
      }

      static inline value_type abs(value_type a) {
        return _mm512_abs_ps(a);
      }

      static inline value_type neg(value_type a) {
        return _mm512_xor_ps(a, _mm512_set1_ps(-0.f));
      }

      static inline value_type rcp(value_type a) {
        return _mm512_rcp14_ps(a);
      }

      static inline value_type exp(value_type a) {
        return exp512_ps(a);
      }

      static inline value_type log(value_type a) {
        return log512_ps(a);
      }

      static inline value_type sin(value_type a) {
        return sin512_ps(a);
      }

      static inline value_type cos(value_type a) {
        return cos512_ps(a);
      }

      static inline value_type tanh(value_type a) {
        return vec_tanh<TARGET_ISA>(a);
      }

      static inline value_type erf(value_type a) {
        return vec_erf<TARGET_ISA>(a);
      }

      static inline value_type max(value_type a, value_type b) {
        return _mm512_max_ps(a, b);
      }

      static inline value_type min(value_type a, value_type b) {
        return _mm512_min_ps(a, b);
      }

      static inline value_type add(value_type a, value_type b) {
        return _mm512_add_ps(a, b);
      }

      static inline value_type sub(value_type a, value_type b) {
        return _mm512_sub_ps(a, b);
      }

      static inline value_type mul(value_type a, value_type b) {
        return _mm512_mul_ps(a, b);
      }

      static inline value_type div(value_type a, value_type b) {
        return _mm512_div_ps(a, b);
      }

      static inline value_type mul_add(value_type a, value_type b, value_type c) {
        return _mm512_fmadd_ps(a, b, c);
      }

      static inline float reduce_add(value_type a) {
        return _mm512_reduce_add_ps(a);
      }

      static inline float reduce_max(value_type a) {
        return _mm512_reduce_max_ps(a);
      }

      static inline value_type round(value_type a) {
          return _mm512_roundscale_ps(a, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
      }

      static inline void convert_and_store(value_type v, int8_t* a, const dim_t count) {
          auto i32 = _mm512_cvttps_epi32(v);
          _mm512_mask_cvtsepi32_storeu_epi8(a,  get_length_mask(count), i32);
      }

      static inline void convert_and_store(value_type v, uint8_t* a, const dim_t count) {
          auto u32 = _mm512_cvttps_epu32(v);
          _mm512_mask_cvtusepi32_storeu_epi8(a,  get_length_mask(count), u32);
      }
    };

  }
}
