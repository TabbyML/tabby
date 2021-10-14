#include "ctranslate2/ops/min_max.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Min::operator()(const StorageView& a, const StorageView& b, StorageView& c) const {
      PROFILE("Min");
      DEVICE_AND_TYPE_DISPATCH(a.device(), a.dtype(), (compute<D, T>(a, b, c)));
    }

    void Max::operator()(const StorageView& a, const StorageView& b, StorageView& c) const {
      PROFILE("Max");
      DEVICE_AND_TYPE_DISPATCH(a.device(), a.dtype(), (compute<D, T>(a, b, c)));
    }
  }
}
