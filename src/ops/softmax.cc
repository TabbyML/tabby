#include "ctranslate2/ops/softmax.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    LogSoftMax::LogSoftMax()
      : SoftMax(/*log=*/true) {
    }

    SoftMax::SoftMax(bool log)
      : _log(log) {
    }

    void SoftMax::operator()(StorageView& x) const {
      operator()(x, nullptr, x);
    }

    void SoftMax::operator()(const StorageView& x, StorageView& y) const {
      operator()(x, nullptr, y);
    }

    void SoftMax::operator()(const StorageView& x, const StorageView& lengths, StorageView& y) const {
      operator()(x, &lengths, y);
    }

    void SoftMax::operator()(const StorageView& x, const StorageView* lengths, StorageView& y) const {
      if (lengths) {
        const dim_t batch_size = x.size() / x.dim(-1);
        if (lengths->size() != batch_size)
          throw std::invalid_argument("Length mask has size "
                                      + std::to_string(lengths->size())
                                      + " which is different than the current batch size "
                                      + std::to_string(batch_size));
      }

      PROFILE(_log ? "LogSoftMax" : "SoftMax");
      y.resize_as(x);
      switch (x.dtype()) {
      case DataType::FLOAT32: {
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
