#include "ctranslate2/ops/mul.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Mul::operator()(const StorageView& a, const StorageView& b, StorageView& c) const {
      PROFILE("Mul");
      DEVICE_AND_TYPE_DISPATCH(a.device(), a.dtype(), (compute<D, T>(a, b, c)));
    }

  }
}
