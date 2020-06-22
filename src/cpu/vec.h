#pragma once

#include <algorithm>
#include <cmath>

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cpu {

    // Interface for vectorized types.
    template <typename T, CpuIsa ISA>
    struct Vec {

      using value_type = T;
      static constexpr dim_t width = 1;

      static value_type load(T value) {
        return value;
      }

      static value_type load(const T* ptr, dim_t count = width) {
        (void)count;
        return *ptr;
      }

      static void store(value_type value, T* ptr, dim_t count = width) {
        (void)count;
        *ptr = value;
      }

      static value_type rcp(value_type a) {
        return static_cast<T>(1) / a;
      }

      static value_type exp(value_type a) {
        return std::exp(a);
      }

      static value_type log(value_type a) {
        return std::log(a);
      }

      static value_type sin(value_type a) {
        return std::sin(a);
      }

      static value_type cos(value_type a) {
        return std::cos(a);
      }

      static value_type max(value_type a, value_type b) {
        return std::max(a, b);
      }

      static value_type min(value_type a, value_type b) {
        return std::min(a, b);
      }

      static value_type add(value_type a, value_type b) {
        return a + b;
      }

      static value_type sub(value_type a, value_type b) {
        return a - b;
      }

      static value_type mul(value_type a, value_type b) {
        return a * b;
      }

    };

    template <typename T, CpuIsa ISA>
    using vec_type = typename Vec<T, ISA>::value_type;

  }
}
