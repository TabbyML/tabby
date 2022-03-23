#pragma once

#include <cstdint>
#include <string>

#include "devices.h"

namespace ctranslate2 {

  using dim_t = int64_t;  // This type should be signed.

  enum class DataType {
    FLOAT,
    INT8,
    INT16,
    INT32,
    FLOAT16
  };

  std::string dtype_name(DataType type);
  bool is_float_type(DataType type);

  enum class ComputeType {
    DEFAULT,
    AUTO,
    FLOAT,
    INT8,
    INT8_FLOAT16,
    INT16,
    FLOAT16
  };

  ComputeType str_to_compute_type(const std::string& compute_type);
  std::string compute_type_to_str(const ComputeType compute_type);

  // Returns the final compute type based on model weights and device information.
  ComputeType resolve_compute_type(const ComputeType requested_compute_type,
                                   const ComputeType model_compute_type,
                                   const Device device,
                                   const int device_index,
                                   const bool enable_fallback = false);

  // Gets the weights data type for the given compute type.
  DataType compute_type_to_data_type(const ComputeType compute_type);

  // Gets the default floating point type for the given compute type.
  DataType get_default_float_type(const ComputeType compute_type);

}
