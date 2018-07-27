#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class ReLU : public UnaryOp {
    public:
      void operator()(StorageView& x) const {
        DEVICE_DISPATCH(x.device(), (compute<D, float>(x)));
      }
      void operator()(const StorageView& x, StorageView& y) const override {
        DEVICE_DISPATCH(x.device(), (compute<D, float>(x, y)));
      }

    private:
      template <Device D, typename T>
      void compute(StorageView& x) const {
        primitives<D>::relu(x.data<T>(), x.size());
      }

      template <Device D, typename T>
      void compute(const StorageView& x, StorageView& y) const {
        y.resize_as(x);
        primitives<D>::relu(x.data<T>(), y.data<T>(), x.size());
      }
    };

  }
}
