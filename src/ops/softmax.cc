#include "ctranslate2/ops/softmax.h"

#include "device_dispatch.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    LogSoftMax::LogSoftMax()
      : SoftMax(/*log=*/true) {
    }

    SoftMax::SoftMax(bool log)
      : _log(log) {
    }

    void SoftMax::operator()(const StorageView& x, StorageView& y) const {
      operator()(x, nullptr, y);
    }

    void SoftMax::operator()(const StorageView& x, const StorageView& lengths, StorageView& y) const {
      operator()(x, &lengths, y);
    }

    void SoftMax::operator()(const StorageView& x, const StorageView* lengths, StorageView& y) const {
      PROFILE(_log ? "LogSoftMax" : "SoftMax");
      y.resize_as(x);
      switch (x.dtype()) {
      case DataType::FLOAT: {
        DEVICE_DISPATCH(x.device(), (compute<D, float>(x, lengths, y)));
        break;
      }
#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16: {
        if (x.device() != Device::CUDA)
          throw std::invalid_argument("FP16 SoftMax is only supported on GPU");
        compute<Device::CUDA, float16_t>(x, lengths, y);
        break;
      }
#endif
      default:
        throw std::invalid_argument("SoftMax only supports float (or float16 on GPU)");
      }
    }

  }
}
