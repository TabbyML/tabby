#include "ctranslate2/ops/gelu.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    GELU::GELU(const Approximation approximation)
      : _approximation(approximation)
    {
    }

    void GELU::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("GELU");

      y.resize_as(x);

      DEVICE_AND_FLOAT_DISPATCH("GELU", x.device(), x.dtype(), (compute<D, T>(x, y)));
    }

  }
}
