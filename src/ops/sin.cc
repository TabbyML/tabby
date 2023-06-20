#include "ctranslate2/ops/sin.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Sin::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("Sin");

      y.resize_as(x);

      switch (x.dtype()) {
      case DataType::FLOAT32: {
        DEVICE_DISPATCH(x.device(), (primitives<D>::sin(x.data<float>(), y.data<float>(), x.size())));
        break;
      }
#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16: {
        if (x.device() != Device::CUDA)
          throw std::invalid_argument("FP16 Sin is only supported on GPU");
        primitives<Device::CUDA>::sin(x.data<float16_t>(), y.data<float16_t>(), x.size());
        break;
      }
#endif
      default:
        throw std::invalid_argument("Sin only supports float types");
      }
    }

  }
}
