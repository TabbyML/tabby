#include "ctranslate2/ops/sub.h"

#include "../device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Sub::operator()(const StorageView& a, const StorageView& b, StorageView& c) const {
      PROFILE_FUN;
      DEVICE_DISPATCH(a.device(), TYPE_DISPATCH(a.dtype(), (compute<D, T>(a, b, c))));
    }

  }
}
