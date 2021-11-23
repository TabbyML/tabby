#pragma once

#include <arm_neon.h>
#include <neon_mathfun.h>

#include "vec.h"

#if defined(__GNUC__) || defined(__clang__)
#  define __ct2_align16__ __attribute__((aligned(16)))
#else
#  define __ct2_align16__
#endif

namespace ctranslate2 {
  namespace cpu {

    template<>
    struct Vec<float, TARGET_ISA> {

      using value_type = float32x4_t;
      static constexpr dim_t width = 4;

      static inline value_type load(float value) {
        return vdupq_n_f32(value);
      }

      static inline value_type load(const float* ptr) {
        return vld1q_f32(ptr);
      }

      static inline value_type load(const float* ptr, dim_t count, float default_value = 0) {
        if (count == width) {
          return vld1q_f32(ptr);
        } else {
          __ct2_align16__ float tmp_values[width];
          std::fill(tmp_values, tmp_values + width, default_value);
          std::copy(ptr, ptr + count, tmp_values);
          return vld1q_f32(tmp_values);
        }
      }

      static inline void store(value_type value, float* ptr) {
        vst1q_f32(ptr, value);
      }

      static inline void store(value_type value, float* ptr, dim_t count) {
        if (count == width) {
          vst1q_f32(ptr, value);
        } else {
          __ct2_align16__ float tmp_values[width];
          vst1q_f32(tmp_values, value);
          std::copy(tmp_values, tmp_values + count, ptr);
        }
      }

      static inline value_type abs(value_type a) {
        return vabsq_f32(a);
      }

      static inline value_type rcp(value_type a) {
        return vrecpeq_f32(a);
      }

      static inline value_type exp(value_type a) {
        return exp_ps(a);
      }

      static inline value_type log(value_type a) {
        return log_ps(a);
      }

      static inline value_type sin(value_type a) {
        return sin_ps(a);
      }

      static inline value_type cos(value_type a) {
        return cos_ps(a);
      }

      static inline value_type max(value_type a, value_type b) {
        return vmaxq_f32(a, b);
      }

      static inline value_type min(value_type a, value_type b) {
        return vminq_f32(a, b);
      }

      static inline value_type add(value_type a, value_type b) {
        return vaddq_f32(a, b);
      }

      static inline value_type sub(value_type a, value_type b) {
        return vsubq_f32(a, b);
      }

      static inline value_type mul(value_type a, value_type b) {
        return vmulq_f32(a, b);
      }

    };

  }
}
