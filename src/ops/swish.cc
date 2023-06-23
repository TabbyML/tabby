#include "ctranslate2/ops/swish.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Swish::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("Swish");
      DEVICE_AND_FLOAT_DISPATCH("Swish", x.device(), x.dtype(), (compute<D, T>(x, y)));
    }

  }
}
