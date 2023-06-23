#include "ctranslate2/ops/cos.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Cos::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("Cos");

      y.resize_as(x);

      DEVICE_AND_FLOAT_DISPATCH("Cos", x.device(), x.dtype(),
                                (primitives<D>::cos(x.data<T>(), y.data<T>(), x.size())));
    }

  }
}
