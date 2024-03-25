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
      using mask_type = uint32x4_t;
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

      static inline value_type load_and_convert(const int32_t* ptr) {
        return vcvtq_f32_s32(vld1q_s32(ptr));
      }

      static inline value_type load_and_convert(const int32_t* ptr,
                                                dim_t count,
                                                int32_t default_value = 0) {
        if (count == width) {
          return load_and_convert(ptr);
        } else {
          __ct2_align16__ int32_t tmp_values[width];
          std::fill(tmp_values, tmp_values + width, default_value);
          std::copy(ptr, ptr + count, tmp_values);
          return load_and_convert(tmp_values);
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

      static inline value_type bit_and(value_type a, value_type b) {
        return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
      }

      static inline value_type bit_xor(value_type a, value_type b) {
        return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
      }

      static inline mask_type lt(value_type a, value_type b) {
        return vcltq_f32(a, b);
      }

      static inline value_type select(mask_type mask, value_type a, value_type b) {
        return vbslq_f32(mask, a, b);
      }

      static inline value_type abs(value_type a) {
        return vabsq_f32(a);
      }

      static inline value_type neg(value_type a) {
        return vnegq_f32(a);
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

      static inline value_type tanh(value_type a) {
        return vec_tanh<TARGET_ISA>(a);
      }

      static inline value_type erf(value_type a) {
        return vec_erf<TARGET_ISA>(a);
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

      static inline value_type div(value_type a, value_type b) {
        return vdivq_f32(a, b);
      }

      static inline value_type mul_add(value_type a, value_type b, value_type c) {
        return vfmaq_f32(c, a, b);
      }

      static inline float reduce_add(value_type a) {
        return vaddvq_f32(a);
      }

      static inline float reduce_max(value_type a) {
        return vmaxvq_f32(a);
      }

      static inline value_type round(value_type v) {
#ifdef __aarch64__
        return vrndiq_f32(v);
#else
        float temp[4] = {std::nearbyintf(v[0]), std::nearbyintf(v[1]), std::nearbyintf(v[2]), std::nearbyintf(v[3])};
        return load(temp);
#endif
      }

      static inline void convert_and_store(value_type v, int8_t *a, dim_t count) {
        //convert float32x4_t to int32x4_t
        auto i32x4 = vcvtq_s32_f32(v);
        //then convert to int16x4_t
        auto i16x4 = vqmovn_s32(i32x4);
        //finally convert to int8x4_t
        auto i8x8 = vqmovn_s16(vcombine_s16(i16x4, vdup_n_s16(0)));
        int8_t tmp[8];
        vst1_s8(tmp, i8x8);
        std::copy(tmp, tmp + count, a);
      }

      static inline void convert_and_store(value_type v, uint8_t *a, dim_t count) {
        //convert float32x4_t to uint32x4_t
        auto u32x4 = vcvtq_u32_f32(v);
        //then convert to uint16x4_t
        auto u16x4 = vqmovn_u32(u32x4);
        //finally convert to uint8x8_t
        auto u8x8 = vqmovn_u16(vcombine_u16(u16x4, vdup_n_u16(0)));
        uint8_t tmp[8];
        vst1_u8(tmp, u8x8);
        std::copy(tmp, tmp + count, a);
      }
  };
}
}
