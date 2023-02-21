#include "ctranslate2/ops/bias_add.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    BiasAdd::BiasAdd(const ActivationType* activation_type)
      : _activation_type(activation_type)
    {
    }

    void BiasAdd::operator()(const StorageView& value,
                             const StorageView& bias,
                             StorageView& output) const {
      PROFILE("BiasAdd");
      output.resize_as(value);
      switch (value.dtype()) {
      case DataType::FLOAT32: {
        DEVICE_DISPATCH(value.device(), (compute<D, float>(value, bias, output)));
        break;
      }
#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16: {
        if (value.device() != Device::CUDA)
          throw std::invalid_argument("FP16 BiasAdd is only supported on GPU");
        compute<Device::CUDA, float16_t>(value, bias, output);
        break;
      }
#endif
      default:
        throw std::invalid_argument("BiasAdd only supports float (or float16 on GPU)");
      }
    }

  }
}
