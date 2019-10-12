#include "ctranslate2/ops/relu.h"

namespace ctranslate2 {
  namespace ops {

    void ReLU::operator()(const StorageView& x, StorageView& y) const {
      DEVICE_DISPATCH(x.device(), (compute<D, float>(x, y)));
    }

  }
}
