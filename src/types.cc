#include "ctranslate2/types.h"

#include <stdexcept>

#include "ctranslate2/utils.h"

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
    if (compute_type == "default")
      return ComputeType::DEFAULT;
    if (compute_type == "auto")
      return ComputeType::AUTO;
    throw std::invalid_argument("Invalid compute type: " + compute_type);
  }

  static inline void unsupported_compute_type(const std::string& name) {
    throw std::invalid_argument("Requested " + name + " compute type, but the target device "
                                "or backend do not support efficient " + name + " computation.");
  }

  ComputeType resolve_compute_type(const ComputeType compute_type,
                                   const DataType weights_type,
                                   const Device device,
                                   const int device_index,
                                   const bool enable_fallback) {
    switch (compute_type) {

    case ComputeType::FLOAT: {
      return ComputeType::FLOAT;
    }

    case ComputeType::FLOAT16: {
      if (mayiuse_float16(device, device_index))
        return ComputeType::FLOAT16;
      if (!enable_fallback)
        unsupported_compute_type("float16");
      return ComputeType::FLOAT;
    }

    case ComputeType::INT16: {
      if (mayiuse_int16(device, device_index))
        return ComputeType::INT16;
      if (!enable_fallback)
        unsupported_compute_type("int16");
      if (device == Device::CPU && mayiuse_int8(device, device_index))
        return ComputeType::INT8;
      if (device == Device::CUDA && mayiuse_float16(device, device_index))
        return ComputeType::FLOAT16;
      return ComputeType::FLOAT;
    }

    case ComputeType::INT8: {
      if (mayiuse_int8(device, device_index))
        return ComputeType::INT8;
      if (!enable_fallback)
        unsupported_compute_type("int8");
      if (device == Device::CPU && mayiuse_int16(device, device_index))
        return ComputeType::INT16;
      if (device == Device::CUDA && mayiuse_float16(device, device_index))
        return ComputeType::FLOAT16;
      return ComputeType::FLOAT;
    }

    case ComputeType::AUTO: {
      if (device == Device::CUDA) {
        if (mayiuse_float16(device, device_index))
          return ComputeType::FLOAT16;
        if (mayiuse_int8(device, device_index))
          return ComputeType::INT8;
      } else {
        if (mayiuse_int8(device, device_index))
          return ComputeType::INT8;
        if (mayiuse_int16(device, device_index))
          return ComputeType::INT16;
      }
      return ComputeType::FLOAT;
    }

    default: {
      // By default, the compute type is the type of the saved model weights.
      // To ensure that any models can be loaded, we enable the fallback.

      ComputeType inferred_compute_type = ComputeType::FLOAT;
      switch (weights_type) {
      case DataType::INT16:
        inferred_compute_type = ComputeType::INT16;
        break;
      case DataType::INT8:
        inferred_compute_type = ComputeType::INT8;
        break;
      case DataType::FLOAT16:
        inferred_compute_type = ComputeType::FLOAT16;
        break;
      default:
        inferred_compute_type = ComputeType::FLOAT;
        break;
      }

      return resolve_compute_type(inferred_compute_type,
                                  weights_type,
                                  device,
                                  device_index,
                                  /*enable_fallback=*/true);
    }

    }
  }

  DataType compute_type_to_data_type(const ComputeType compute_type) {
    switch (compute_type) {
    case ComputeType::FLOAT:
      return DataType::FLOAT;
    case ComputeType::INT8:
      return DataType::INT8;
    case ComputeType::INT16:
      return DataType::INT16;
    case ComputeType::FLOAT16:
      return DataType::FLOAT16;
    default:
      throw std::invalid_argument("resolve_compute_type should be called first");
    }
  }

}
