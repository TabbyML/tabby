#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Transpose : public UnaryOp {
    public:
      Transpose() = default;
      Transpose(const std::vector<dim_t>& perm);
      void operator()(const StorageView& x, StorageView& y) const override;

    private:
      std::vector<dim_t> _perm;

      template <Device D, typename T>
      void compute(const StorageView& x, const std::vector<dim_t>& perm, StorageView& y) const {
        if (x.rank() == 2) {
          y.resize({x.dim(1), x.dim(0)});
          primitives<D>::transpose_2d(x.data<T>(), x.shape().data(), y.data<T>());
        } else if (x.rank() == 3) {
          y.resize({x.dim(perm[0]), x.dim(perm[1]), x.dim(perm[2])});
          primitives<D>::transpose_3d(x.data<T>(), x.shape().data(), perm.data(), y.data<T>());
        } else if (x.rank() == 4) {
          y.resize({x.dim(perm[0]), x.dim(perm[1]), x.dim(perm[2]), x.dim(perm[3])});
          primitives<D>::transpose_4d(x.data<T>(), x.shape().data(), perm.data(), y.data<T>());
        } else {
          throw std::invalid_argument("Transpose: rank " + std::to_string(x.rank())
                                      + " is not supported, supported ranks are: 2, 3, 4");
        }
      }
    };

  }
}
