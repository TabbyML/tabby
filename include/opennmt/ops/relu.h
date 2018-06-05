#pragma once

#include "opennmt/ops/op.h"

namespace opennmt {
  namespace ops {

    class ReLU : public UnaryOp {
    public:
      void operator()(StorageView& x) const {
        TYPE_DISPATCH(x.dtype(), compute<T>(x));
      }
      void operator()(const StorageView& x, StorageView& y) const override {
        TYPE_DISPATCH(x.dtype(), compute<T>(x, y));
      }

    private:
      template <typename T>
      void compute(StorageView& x) const {
        primitives::relu(x.data<T>(), x.size());
      }

      template <typename T>
      void compute(const StorageView& x, StorageView& y) const {
        y.resize_as(x);
        primitives::relu(x.data<T>(), y.data<T>(), x.size());
      }
    };

  }
}
