#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace ctranslate2 {

  // Adapted from https://github.com/oneapi-src/oneDNN/blob/v3.0.1/src/common/bfloat16.hpp

  template <typename T, typename U>
  inline T bit_cast(const U &u) {
    T t;
    std::memcpy(&t, &u, sizeof(U));
    return t;
  }

  class bfloat16_t {
  public:
    bfloat16_t() = default;
    bfloat16_t(float f) {
      *this = f;
    }

    bfloat16_t& operator=(float f) {
      auto iraw = bit_cast<std::array<uint16_t, 2>>(f);
      switch (std::fpclassify(f)) {
      case FP_SUBNORMAL:
      case FP_ZERO:
        // sign preserving zero (denormal go to zero)
        _bits = iraw[1];
        _bits &= 0x8000;
        break;
      case FP_INFINITE:
        _bits = iraw[1];
        break;
      case FP_NAN:
        // truncate and set MSB of the mantissa force QNAN
        _bits = iraw[1];
        _bits |= 1 << 6;
        break;
      case FP_NORMAL:
        // round to nearest even and truncate
        const uint32_t rounding_bias = 0x00007FFF + (iraw[1] & 0x1);
        const uint32_t int_raw = bit_cast<uint32_t>(f) + rounding_bias;
        iraw = bit_cast<std::array<uint16_t, 2>>(int_raw);
        _bits = iraw[1];
        break;
      }

      return *this;
    }

    operator float() const {
      std::array<uint16_t, 2> iraw = {{0, _bits}};
      return bit_cast<float>(iraw);
    }

  private:
    uint16_t _bits;

    // Converts the 32 bits of a normal float or zero to the bits of a bfloat16.
    static constexpr uint16_t convert_bits_of_normal_or_zero(const uint32_t bits) {
      return uint32_t{bits + uint32_t{0x7FFFU + (uint32_t{bits >> 16} & 1U)}} >> 16;
    }
  };

}
