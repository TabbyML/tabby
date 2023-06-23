#include "ctranslate2/ops/tanh.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Tanh::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("Tanh");

      y.resize_as(x);

      DEVICE_AND_FLOAT_DISPATCH("Tanh", x.device(), x.dtype(),
                                (primitives<D>::tanh(x.data<T>(), y.data<T>(), x.size())));
    }

  }
}
