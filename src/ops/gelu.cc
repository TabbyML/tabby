#include "ctranslate2/ops/gelu.h"

#include "../device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void GELU::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("GELU");
      DEVICE_DISPATCH(x.device(), (compute<D, float>(x, y)));
    }

  }
}
