#include "ctranslate2/ops/swish.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Swish::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("Swish");
      switch (x.dtype()) {
      case DataType::FLOAT32: {
        DEVICE_DISPATCH(x.device(), (compute<D, float>(x, y)));
        break;
      }
#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16: {
        if (x.device() != Device::CUDA)
          throw std::invalid_argument("FP16 Swish is only supported on GPU");
        compute<Device::CUDA, float16_t>(x, y);
        break;
      }
#endif
      default:
        throw std::invalid_argument("Swish only supports float (or float16 on GPU)");
      }
    }

  }
}
