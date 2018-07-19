#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Transpose : public UnaryOp {
    public:
      Transpose() = default;
      Transpose(const std::vector<size_t>& perm)
        : _perm(perm) {
      }

      void operator()(const StorageView& x, StorageView& y) const override {
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

        TYPE_DISPATCH(x.dtype(), compute<T>(x, perm, y));
      }

    private:
      std::vector<size_t> _perm;

      template <typename T>
      void compute(const StorageView& x, const std::vector<size_t>& perm, StorageView& y) const {
        if (x.rank() == 2) {
          y.resize({x.dim(1), x.dim(0)});
          primitives::transpose_2d(x.data<T>(), x.shape().data(), y.data<T>());
        } else if (x.rank() == 3) {
          y.resize({x.dim(perm[0]), x.dim(perm[1]), x.dim(perm[2])});
          primitives::transpose_3d(x.data<T>(), x.shape().data(), perm.data(), y.data<T>());
        } else if (x.rank() == 4) {
          y.resize({x.dim(perm[0]), x.dim(perm[1]), x.dim(perm[2]), x.dim(perm[3])});
          primitives::transpose_4d(x.data<T>(), x.shape().data(), perm.data(), y.data<T>());
        } else {
          throw std::invalid_argument("unsupported rank for transposition");
        }
      }
    };

  }
}
