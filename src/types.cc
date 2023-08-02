#include "ctranslate2/types.h"

#include <stdexcept>

#ifdef CT2_WITH_CUDA
#  include "./cuda/utils.h"
#endif

#include "cpu/backend.h"
#include "env.h"

namespace ctranslate2 {

  std::string dtype_name(DataType type) {
    switch (type) {
    case DataType::FLOAT32:
      return "float32";
    case DataType::INT8:
      return "int8";
    case DataType::INT16:
      return "int16";
    case DataType::INT32:
      return "int32";
    case DataType::FLOAT16:
      return "float16";
    case DataType::BFLOAT16:
      return "bfloat16";
    default:
      return "";
    }
  }

  bool is_float_type(DataType type) {
    switch (type) {
    case DataType::FLOAT32:
    case DataType::FLOAT16:
    case DataType::BFLOAT16:
      return true;
    default:
      return false;
    }
  }

  ComputeType str_to_compute_type(const std::string& compute_type) {
    if (compute_type == "int8")
      return ComputeType::INT8;
    if (compute_type == "int8_float32")
      return ComputeType::INT8_FLOAT32;
    if (compute_type == "int8_float16")
      return ComputeType::INT8_FLOAT16;
    if (compute_type == "int8_bfloat16")
      return ComputeType::INT8_BFLOAT16;
    if (compute_type == "int16")
      return ComputeType::INT16;
    if (compute_type == "float32" || compute_type == "float")
      return ComputeType::FLOAT32;
    if (compute_type == "float16")
      return ComputeType::FLOAT16;
    if (compute_type == "bfloat16")
      return ComputeType::BFLOAT16;
    if (compute_type == "default")
      return ComputeType::DEFAULT;
    if (compute_type == "auto")
      return ComputeType::AUTO;
    throw std::invalid_argument("Invalid compute type: " + compute_type);
  }

  std::string compute_type_to_str(const ComputeType compute_type) {
    switch (compute_type) {
    case ComputeType::DEFAULT:
      return "default";
    case ComputeType::AUTO:
      return "auto";
    case ComputeType::FLOAT32:
      return "float32";
    case ComputeType::INT8:
      return "int8";
    case ComputeType::INT8_FLOAT32:
      return "int8_float32";
    case ComputeType::INT8_FLOAT16:
      return "int8_float16";
    case ComputeType::INT8_BFLOAT16:
      return "int8_bfloat16";
    case ComputeType::INT16:
      return "int16";
    case ComputeType::FLOAT16:
      return "float16";
    case ComputeType::BFLOAT16:
      return "bfloat16";
    };
    throw std::invalid_argument("Invalid compute type value");
  }

  bool mayiuse_bfloat16(const Device device, const int device_index) {
    switch (device) {
    case Device::CUDA: {
#ifdef CT2_WITH_CUDA
      static const bool allow_bfloat16 = read_bool_from_env("CT2_CUDA_ALLOW_BF16");
      return allow_bfloat16 || cuda::get_device_properties(device_index).major >= 8;
#else
      (void)device_index;
      return false;
#endif
    }
    default:
      return false;
    }
  }

  bool mayiuse_float16(const Device device, const int device_index) {
    switch (device) {
    case Device::CUDA: {
#ifdef CT2_WITH_CUDA
      static const bool allow_float16 = read_bool_from_env("CT2_CUDA_ALLOW_FP16");
      return allow_float16 || cuda::gpu_has_fp16_tensor_cores(device_index);
#else
      (void)device_index;
      return false;
#endif
    }
    default:
      return false;
    }
  }

  bool mayiuse_int16(const Device device, const int) {
    switch (device) {
    case Device::CPU:
      return cpu::has_gemm_backend(ComputeType::INT16);
    default:
      return false;
    }
  }

  bool mayiuse_int8(const Device device, const int device_index) {
    switch (device) {
    case Device::CUDA:
#ifdef CT2_WITH_CUDA
      return cuda::gpu_supports_int8(device_index);
#else
      (void)device_index;
      return false;
#endif
    case Device::CPU:
      return cpu::has_gemm_backend(ComputeType::INT8);
    default:
      return false;
    }
  }

  static inline void unsupported_compute_type(const std::string& name) {
    throw std::invalid_argument("Requested " + name + " compute type, but the target device "
                                "or backend do not support efficient " + name + " computation.");
  }

  ComputeType resolve_compute_type(const ComputeType requested_compute_type,
                                   const ComputeType model_compute_type,
                                   const Device device,
                                   const int device_index,
                                   const bool enable_fallback) {
    const bool support_bfloat16 = mayiuse_bfloat16(device, device_index);
    const bool support_float16 = mayiuse_float16(device, device_index);
    const bool support_int16 = mayiuse_int16(device, device_index);
    const bool support_int8 = mayiuse_int8(device, device_index);

    switch (requested_compute_type) {

    case ComputeType::FLOAT32: {
      return ComputeType::FLOAT32;
    }

    case ComputeType::FLOAT16: {
      if (support_float16)
        return ComputeType::FLOAT16;
      if (!enable_fallback)
        unsupported_compute_type("float16");
      return ComputeType::FLOAT32;
    }

    case ComputeType::BFLOAT16: {
      if (support_bfloat16)
        return ComputeType::BFLOAT16;
      if (!enable_fallback)
        unsupported_compute_type("bfloat16");
      return ComputeType::FLOAT32;
    }

    case ComputeType::INT16: {
      if (support_int16)
        return ComputeType::INT16;
      if (!enable_fallback)
        unsupported_compute_type("int16");
      if (device == Device::CPU && support_int8)
        return ComputeType::INT8_FLOAT32;
      if (device == Device::CUDA && support_float16)
        return ComputeType::FLOAT16;
      return ComputeType::FLOAT32;
    }

    case ComputeType::INT8: {
      const DataType float_type = compute_type_to_data_type(model_compute_type).second;
      ComputeType actual_compute_type = ComputeType::INT8_FLOAT32;

      switch (float_type) {
      case DataType::FLOAT16:
        actual_compute_type = ComputeType::INT8_FLOAT16;
        break;
      case DataType::BFLOAT16:
        actual_compute_type = ComputeType::INT8_BFLOAT16;
        break;
      default:
        actual_compute_type = ComputeType::INT8_FLOAT32;
        break;
      }

      const ComputeType resolved_compute_type = resolve_compute_type(actual_compute_type,
                                                                     model_compute_type,
                                                                     device,
                                                                     device_index,
                                                                     /*enable_fallback=*/true);

      const DataType weight_type = compute_type_to_data_type(resolved_compute_type).first;
      if (weight_type != DataType::INT8)
        unsupported_compute_type("int8");

      return resolved_compute_type;
    }

    case ComputeType::INT8_FLOAT32: {
      if (support_int8)
        return ComputeType::INT8_FLOAT32;
      if (!enable_fallback)
        unsupported_compute_type("int8_float32");
      if (device == Device::CPU && support_int16)
        return ComputeType::INT16;
      if (device == Device::CUDA && support_float16)
        return ComputeType::FLOAT16;
      return ComputeType::FLOAT32;
    }

    case ComputeType::INT8_FLOAT16: {
      if (support_int8 && support_float16)
        return ComputeType::INT8_FLOAT16;
      if (!enable_fallback)
        unsupported_compute_type("int8_float16");
      if (support_int8)
        return ComputeType::INT8_FLOAT32;
      if (support_float16)
        return ComputeType::FLOAT16;
      if (support_int16)
        return ComputeType::INT16;
      return ComputeType::FLOAT32;
    }

    case ComputeType::INT8_BFLOAT16: {
      if (support_int8 && support_bfloat16)
        return ComputeType::INT8_BFLOAT16;
      if (!enable_fallback)
        unsupported_compute_type("int8_bfloat16");
      if (support_int8)
        return ComputeType::INT8_FLOAT32;
      if (support_bfloat16)
        return ComputeType::BFLOAT16;
      return ComputeType::FLOAT32;
    }

    case ComputeType::AUTO: {
      if (device == Device::CUDA) {
        if (support_int8 && support_float16)
          return ComputeType::INT8_FLOAT16;
        if (support_int8)
          return ComputeType::INT8_FLOAT32;
        if (support_float16)
          return ComputeType::FLOAT16;
      } else {
        if (support_int8)
          return ComputeType::INT8_FLOAT32;
        if (support_int16)
          return ComputeType::INT16;
      }
      return ComputeType::FLOAT32;
    }

    default: {
      // By default, the compute type is the type of the saved model weights.
      // To ensure that any models can be loaded, we enable the fallback.
      return resolve_compute_type(model_compute_type,
                                  model_compute_type,
                                  device,
                                  device_index,
                                  /*enable_fallback=*/true);
    }

    }
  }

  std::pair<DataType, DataType> compute_type_to_data_type(const ComputeType compute_type) {
    switch (compute_type) {
    case ComputeType::FLOAT32:
      return std::make_pair(DataType::FLOAT32, DataType::FLOAT32);
    case ComputeType::INT8_FLOAT32:
      return std::make_pair(DataType::INT8, DataType::FLOAT32);
    case ComputeType::INT8_FLOAT16:
      return std::make_pair(DataType::INT8, DataType::FLOAT16);
    case ComputeType::INT8_BFLOAT16:
      return std::make_pair(DataType::INT8, DataType::BFLOAT16);
    case ComputeType::INT16:
      return std::make_pair(DataType::INT16, DataType::FLOAT32);
    case ComputeType::FLOAT16:
      return std::make_pair(DataType::FLOAT16, DataType::FLOAT16);
    case ComputeType::BFLOAT16:
      return std::make_pair(DataType::BFLOAT16, DataType::BFLOAT16);
    default:
      throw std::invalid_argument("resolve_compute_type should be called first");
    }
  }

  ComputeType data_type_to_compute_type(const DataType weight_type, const DataType float_type) {
    switch (weight_type) {
    case DataType::INT8: {
      switch (float_type) {
      case DataType::FLOAT16:
        return ComputeType::INT8_FLOAT16;
      case DataType::BFLOAT16:
        return ComputeType::INT8_BFLOAT16;
      default:
        return ComputeType::INT8_FLOAT32;
      }
    }
    case DataType::INT16:
      return ComputeType::INT16;
    case DataType::FLOAT16:
      return ComputeType::FLOAT16;
    case DataType::BFLOAT16:
      return ComputeType::BFLOAT16;
    default:
      return ComputeType::FLOAT32;
    }
  }

  DataType get_default_float_type(const ComputeType compute_type) {
    return compute_type_to_data_type(compute_type).second;
  }

  dim_t get_preferred_size_multiple(const ComputeType compute_type,
                                    const Device device,
                                    const int device_index) {
#ifdef CT2_WITH_CUDA
    if (device == Device::CUDA) {
      if ((compute_type == ComputeType::FLOAT16 || compute_type == ComputeType::BFLOAT16)
          && cuda::gpu_has_fp16_tensor_cores(device_index))
        return 8;
    }
#else
    (void)compute_type;
    (void)device;
    (void)device_index;
#endif
    return 1;
  }

}
