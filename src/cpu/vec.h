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
      static constexpr dim_t width = 1;

      static inline value_type load(T value) {
        return value;
      }

      static inline value_type load(const T* ptr, dim_t count = width, T default_value = 0) {
        (void)count;
        (void)default_value;
        return *ptr;
      }

      static inline void store(value_type value, T* ptr, dim_t count = width) {
        (void)count;
        *ptr = value;
      }

      static inline value_type abs(value_type a) {
        return std::abs(a);
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

    };

    template <typename T, CpuIsa ISA = CpuIsa::GENERIC>
    using vec_type = typename Vec<T, ISA>::value_type;

  }
}
