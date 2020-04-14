#include "ctranslate2/ops/min_max.h"

#include "../device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Min::operator()(const StorageView& a, const StorageView& b, StorageView& c) const {
      PROFILE("Min");
      DEVICE_DISPATCH(a.device(), TYPE_DISPATCH(a.dtype(), (compute<D, T>(a, b, c))));
    }

    void Max::operator()(const StorageView& a, const StorageView& b, StorageView& c) const {
      PROFILE("Max");
      DEVICE_DISPATCH(a.device(), TYPE_DISPATCH(a.dtype(), (compute<D, T>(a, b, c))));
    }
  }
}
