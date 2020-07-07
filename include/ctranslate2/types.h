#pragma once

#include <cstdint>
#include <string>

namespace ctranslate2 {

  using dim_t = int64_t;  // This type should be signed.

  enum class DataType {
    FLOAT,
    INT8,
    INT16,
    INT32
  };

  std::string dtype_name(DataType type);

  enum class ComputeType {
    DEFAULT,
    FLOAT,
    INT8,
    INT16
  };

  ComputeType str_to_compute_type(const std::string& compute_type);

}
