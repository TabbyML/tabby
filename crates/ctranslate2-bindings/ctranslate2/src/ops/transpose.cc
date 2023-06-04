#include "ctranslate2/ops/transpose.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    Transpose::Transpose(const std::vector<dim_t>& perm)
      : _perm(perm) {
    }

    void Transpose::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("Transpose");
      if (x.rank() <= 1) {
        y = x;
        return;
      }

      std::vector<dim_t> perm;
      bool identity = true;
      if (_perm.empty()) {
        perm.resize(x.rank());
        for (dim_t i = 0; i < x.rank(); ++i)
          perm[i] = x.rank() - i - 1;
        identity = false;
      } else {
        assert(_perm.size() == x.rank());
        perm = _perm;
        for (dim_t i = 0; i < x.rank(); ++i) {
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

      DEVICE_AND_TYPE_DISPATCH(x.device(), x.dtype(), (compute<D, T>(x, perm, y)));
    }

  }
}
