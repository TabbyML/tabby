#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cpu {

    // Interface for vectorized types.
    template <typename T, CpuIsa ISA = CpuIsa::GENERIC>
    struct Vec {

      using value_type = T;
      using mask_type = bool;
      static constexpr dim_t width = 1;

      static inline value_type load(T value) {
        return value;
      }

      static inline value_type load(const T* ptr) {
        return *ptr;
      }

      static inline value_type load(const T* ptr, dim_t count, T default_value = T(0)) {
        (void)count;
        (void)default_value;
        return *ptr;
      }

      static inline void store(value_type value, T* ptr) {
        *ptr = value;
      }

      static inline void store(value_type value, T* ptr, dim_t count) {
        (void)count;
        *ptr = value;
      }

      static inline mask_type lt(value_type a, value_type b) {
        return a < b;
      }

      static inline value_type select(mask_type mask, value_type a, value_type b) {
        return mask ? a : b;
      }

      static inline value_type abs(value_type a) {
        return static_cast<value_type>(std::abs(a));
      }

      static inline value_type neg(value_type a) {
        return -a;
      }

      static inline value_type rcp(value_type a) {
        return static_cast<T>(1) / a;
      }

      static inline value_type exp(value_type a) {
        return std::exp(a);
      }

      static inline value_type log(value_type a) {
        return std::log(a);
      }

      static inline value_type sin(value_type a) {
        return std::sin(a);
      }

      static inline value_type cos(value_type a) {
        return std::cos(a);
      }

      static inline value_type tanh(value_type a) {
        return std::tanh(a);
      }

      static inline value_type max(value_type a, value_type b) {
        return std::max(a, b);
      }

      static inline value_type min(value_type a, value_type b) {
        return std::min(a, b);
      }

      static inline value_type add(value_type a, value_type b) {
        return a + b;
      }

      static inline value_type sub(value_type a, value_type b) {
        return a - b;
      }

      static inline value_type mul(value_type a, value_type b) {
        return a * b;
      }

      static inline value_type div(value_type a, value_type b) {
        return a / b;
      }

      static inline value_type mul_add(value_type a, value_type b, value_type c) {
        return a * b + c;
      }

      static inline T reduce_add(value_type a) {
        return a;
      }

      static inline T reduce_max(value_type a) {
        return a;
      }

    };

    template <typename T, CpuIsa ISA = CpuIsa::GENERIC>
    using vec_type = typename Vec<T, ISA>::value_type;

    template <CpuIsa ISA>
    vec_type<float, ISA> vec_tanh(vec_type<float, ISA> a) {
      using VecType = Vec<float, ISA>;

      // Implementation ported from Eigen:
      // https://gitlab.com/libeigen/eigen/-/blob/3.4.0/Eigen/src/Core/MathFunctionsImpl.h#L18-L76

      const auto plus_clamp = VecType::load(7.90531110763549805f);
      const auto minus_clamp = VecType::load(-7.90531110763549805f);
      const auto tiny = VecType::load(0.0004f);
      const auto x = VecType::max(VecType::min(a, plus_clamp), minus_clamp);
      const auto tiny_mask = VecType::lt(VecType::abs(a), tiny);

      const auto alpha_1 = VecType::load(4.89352455891786e-03f);
      const auto alpha_3 = VecType::load(6.37261928875436e-04f);
      const auto alpha_5 = VecType::load(1.48572235717979e-05f);
      const auto alpha_7 = VecType::load(5.12229709037114e-08f);
      const auto alpha_9 = VecType::load(-8.60467152213735e-11f);
      const auto alpha_11 = VecType::load(2.00018790482477e-13f);
      const auto alpha_13 = VecType::load(-2.76076847742355e-16f);

      const auto beta_0 = VecType::load(4.89352518554385e-03f);
      const auto beta_2 = VecType::load(2.26843463243900e-03f);
      const auto beta_4 = VecType::load(1.18534705686654e-04f);
      const auto beta_6 = VecType::load(1.19825839466702e-06f);

      const auto x2 = VecType::mul(x, x);

      auto p = VecType::mul_add(x2, alpha_13, alpha_11);
      p = VecType::mul_add(x2, p, alpha_9);
      p = VecType::mul_add(x2, p, alpha_7);
      p = VecType::mul_add(x2, p, alpha_5);
      p = VecType::mul_add(x2, p, alpha_3);
      p = VecType::mul_add(x2, p, alpha_1);
      p = VecType::mul(x, p);

      auto q = VecType::mul_add(x2, beta_6, beta_4);
      q = VecType::mul_add(x2, q, beta_2);
      q = VecType::mul_add(x2, q, beta_0);

      return VecType::select(tiny_mask, x, VecType::div(p, q));
    }

  }
}
