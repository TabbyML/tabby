#include "ctranslate2/ops/transpose.h"

#include "device_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    Transpose::Transpose(const std::vector<size_t>& perm)
      : _perm(perm) {
    }

    void Transpose::operator()(const StorageView& x, StorageView& y) const {
      if (x.rank() == 1) {
        y = x;
        return;
      }

      std::vector<size_t> perm;
      bool identity = true;
      if (_perm.empty()) {
        perm.resize(x.rank());
        for (size_t i = 0; i < x.rank(); ++i)
          perm[i] = x.rank() - i - 1;
        identity = false;
      } else {
        assert(_perm.size() == x.rank());
        perm = _perm;
        for (size_t i = 0; i < x.rank(); ++i) {
          if (perm[i] != i) {
            identity = false;
            break;
          }
        }
      }

      if (identity) {
        y = x;
        return;
      }

      DEVICE_DISPATCH(x.device(), TYPE_DISPATCH(x.dtype(), (compute<D, T>(x, perm, y))));
    }

  }
}
