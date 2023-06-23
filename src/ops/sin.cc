#include "ctranslate2/ops/sin.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Sin::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("Sin");

      y.resize_as(x);

      DEVICE_AND_FLOAT_DISPATCH("Sin", x.device(), x.dtype(),
                                (primitives<D>::sin(x.data<T>(), y.data<T>(), x.size())));
    }

  }
}
