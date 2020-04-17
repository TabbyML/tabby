#include "ctranslate2/ops/log.h"

#include "../device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Log::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("Log");
      DEVICE_DISPATCH(x.device(), (compute<D, float>(x, y)));
    }

  }
}
