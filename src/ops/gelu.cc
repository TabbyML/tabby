#include "ctranslate2/ops/gelu.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    GELU::GELU(const Approximation approximation)
      : _approximation(approximation)
    {
    }

    void GELU::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("GELU");

      y.resize_as(x);

      switch (x.dtype()) {
      case DataType::FLOAT32: {
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
