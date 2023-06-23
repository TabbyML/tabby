#include "ctranslate2/ops/log.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Log::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("Log");
      DEVICE_AND_FLOAT_DISPATCH("Log", x.device(), x.dtype(), (compute<D, T>(x, y)));
    }

  }
}
