#pragma once

#include <cstdint>
#include <string>

#include "devices.h"
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

  // Returns the final compute type based on model weights and device information.
  ComputeType resolve_compute_type(const ComputeType compute_type,
                                   const DataType weights_type,
                                   const Device device,
                                   const int device_index,
                                   const bool enable_fallback = false);

  // Gets the weights data type for the given compute type.
  DataType compute_type_to_data_type(const ComputeType compute_type);

}
