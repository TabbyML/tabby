#include "ctranslate2/ops/tanh.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Tanh::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("Tanh");

      y.resize_as(x);

      switch (x.dtype()) {
      case DataType::FLOAT32: {
        DEVICE_DISPATCH(x.device(), (primitives<D>::tanh(x.data<float>(), y.data<float>(), x.size())));
        break;
      }
#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16: {
        if (x.device() != Device::CUDA)
          throw std::invalid_argument("FP16 Tanh is only supported on GPU");
        primitives<Device::CUDA>::tanh(x.data<float16_t>(), y.data<float16_t>(), x.size());
        break;
      }
#endif
      default:
        throw std::invalid_argument("Tanh only supports float (or float16 on GPU)");
      }
    }

  }
}
