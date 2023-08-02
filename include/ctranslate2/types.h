#pragma once

#include <cstdint>
#include <string>

#include <half_float/half.hpp>

#include "bfloat16.h"
#include "devices.h"

namespace ctranslate2 {

  using dim_t = int64_t;  // This type should be signed.
  using float16_t = half_float::half;

  enum class DataType {
    FLOAT32,
    INT8,
    INT16,
    INT32,
    FLOAT16,
    BFLOAT16,
  };

  std::string dtype_name(DataType type);
  bool is_float_type(DataType type);

  enum class ComputeType {
    DEFAULT,
    AUTO,
    FLOAT32,
    INT8,
    INT8_FLOAT32,
    INT8_FLOAT16,
    INT8_BFLOAT16,
    INT16,
    FLOAT16,
    BFLOAT16,
  };

  ComputeType str_to_compute_type(const std::string& compute_type);
  std::string compute_type_to_str(const ComputeType compute_type);

  bool mayiuse_bfloat16(const Device device, const int device_index = 0);
  bool mayiuse_float16(const Device device, const int device_index = 0);
  bool mayiuse_int16(const Device device, const int device_index = 0);
  bool mayiuse_int8(const Device device, const int device_index = 0);

  // Returns the final compute type based on model weights and device information.
  ComputeType resolve_compute_type(const ComputeType requested_compute_type,
                                   const ComputeType model_compute_type,
                                   const Device device,
                                   const int device_index = 0,
                                   const bool enable_fallback = false);

  // Maps the compute type from or to the weights data type.
  ComputeType data_type_to_compute_type(const DataType weight_type, const DataType float_type);
  std::pair<DataType, DataType> compute_type_to_data_type(const ComputeType compute_type);

  // Gets the default floating point type for the given compute type.
  DataType get_default_float_type(const ComputeType compute_type);

  // Gets the preferred size multiple for this compute type.
  dim_t get_preferred_size_multiple(const ComputeType compute_type,
                                    const Device device,
                                    const int device_index = 0);

}
