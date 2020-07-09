#pragma once

#include <cstdint>
#include <string>

#include "half_float/half.hpp"

namespace ctranslate2 {

  using dim_t = int64_t;  // This type should be signed.
  using float16_t = half_float::half;

  enum class DataType {
    FLOAT,
    INT8,
    INT16,
    INT32,
    FLOAT16
  };

  std::string dtype_name(DataType type);

  enum class ComputeType {
    DEFAULT,
    FLOAT,
    INT8,
    INT16,
    FLOAT16
  };

  ComputeType str_to_compute_type(const std::string& compute_type);

}
