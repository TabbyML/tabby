#include "ctranslate2/types.h"

namespace ctranslate2 {

  std::string dtype_name(DataType type) {
    switch (type) {
    case DataType::FLOAT:
      return "float";
    case DataType::INT8:
      return "int8";
    case DataType::INT16:
      return "int16";
    case DataType::INT32:
      return "int32";
    case DataType::FLOAT16:
      return "float16";
    default:
      return "";
    }
  }

  ComputeType str_to_compute_type(const std::string& compute_type) {
    if (compute_type == "int8")
      return ComputeType::INT8;
    if (compute_type == "int16")
      return ComputeType::INT16;
    if (compute_type == "float")
      return ComputeType::FLOAT;
    if (compute_type == "float16")
      return ComputeType::FLOAT16;
    return ComputeType::DEFAULT;
  }

}
