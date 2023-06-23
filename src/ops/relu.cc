#include "ctranslate2/ops/relu.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void ReLU::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("ReLU");
      DEVICE_AND_FLOAT_DISPATCH("ReLU", x.device(), x.dtype(), (compute<D, T>(x, y)));
    }

  }
}
