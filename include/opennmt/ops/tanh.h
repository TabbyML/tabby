#pragma once

#include "opennmt/ops/op.h"

namespace opennmt {
  namespace ops {

    class Tanh : public UnaryOp {
    public:
      void operator()(const StorageView& x, StorageView& y) const override {
        compute<float>(x, y);
      }

    private:
      template <typename T>
      void compute(const StorageView& x, StorageView& y) const {
        y.resize_as(x);
        primitives::tanh(x.data<T>(), y.data<T>(), x.size());
      }
    };

  }
}
