#include "ctranslate2/ops/softmax.h"

#include "../device_dispatch.h"

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
      PROFILE_FUN;
      if (lengths && lengths->dim(0) == 1)  // Disable masking when batch size is 1.
        lengths = nullptr;
      y.resize_as(x);
      DEVICE_DISPATCH(x.device(), (compute<D, float>(x, lengths, y)));
    }

  }
}
