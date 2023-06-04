#include "ctranslate2/ops/relu.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void ReLU::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("ReLU");
      switch (x.dtype()) {
      case DataType::FLOAT32: {
        DEVICE_DISPATCH(x.device(), (compute<D, float>(x, y)));
        break;
      }
#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16: {
        if (x.device() != Device::CUDA)
          throw std::invalid_argument("FP16 ReLU is only supported on GPU");
        compute<Device::CUDA, float16_t>(x, y);
        break;
      }
#endif
      default:
        throw std::invalid_argument("ReLU only supports float (or float16 on GPU)");
      }
    }

  }
}
