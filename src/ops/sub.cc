#include "ctranslate2/ops/sub.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Sub::operator()(const StorageView& a, const StorageView& b, StorageView& c) const {
      PROFILE("Sub");
      DEVICE_AND_TYPE_DISPATCH(a.device(), a.dtype(), (compute<D, T>(a, b, c)));
    }

  }
}
