#include "ctranslate2/ops/add.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void Add::operator()(const StorageView& a, const StorageView& b, StorageView& c) const {
      PROFILE("Add");
      DEVICE_AND_TYPE_DISPATCH(a.device(), a.dtype(), (compute<D, T>(a, b, c)));
    }

  }
}
