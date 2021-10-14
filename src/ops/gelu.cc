#include "ctranslate2/ops/gelu.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void GELU::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("GELU");
      switch (x.dtype()) {
      case DataType::FLOAT: {
        DEVICE_DISPATCH(x.device(), (compute<D, float>(x, y)));
        break;
      }
#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16: {
        if (x.device() != Device::CUDA)
          throw std::invalid_argument("FP16 GELU is only supported on GPU");
        compute<Device::CUDA, float16_t>(x, y);
        break;
      }
#endif
      default:
        throw std::invalid_argument("GELU only supports float (or float16 on GPU)");
      }
    }

  }
}
