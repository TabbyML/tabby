#include "ctranslate2/ops/mul.h"

#include "device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Mul::operator()(const StorageView& a, const StorageView& b, StorageView& c) const {
      DEVICE_DISPATCH(a.device(), TYPE_DISPATCH(a.dtype(), (compute<D, T>(a, b, c))));
    }

  }
}
