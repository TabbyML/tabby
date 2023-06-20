#include "ctranslate2/ops/cos.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Cos::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("Cos");

      y.resize_as(x);

      switch (x.dtype()) {
      case DataType::FLOAT32: {
        DEVICE_DISPATCH(x.device(), (primitives<D>::cos(x.data<float>(), y.data<float>(), x.size())));
        break;
      }
#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16: {
        if (x.device() != Device::CUDA)
          throw std::invalid_argument("FP16 Cos is only supported on GPU");
        primitives<Device::CUDA>::cos(x.data<float16_t>(), y.data<float16_t>(), x.size());
        break;
      }
#endif
      default:
        throw std::invalid_argument("Cos only supports float types");
      }
    }

  }
}
