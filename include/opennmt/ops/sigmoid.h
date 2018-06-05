#pragma once

#include "opennmt/ops/op.h"

namespace opennmt {
  namespace ops {

    class Sigmoid : public UnaryOp {
    public:
      void operator()(const StorageView& x, StorageView& y) const override {
        compute<float>(x, y);
      }

    private:
      template <typename T>
      void compute(const StorageView& x, StorageView& y) const {
        y = x;
        compute::mul(static_cast<T>(-1), y.data<T>(), y.size());
        compute::exp(y.data<T>(), y.data<T>(), y.size());
        compute::add(static_cast<T>(1), y.data<T>(), y.size());
        compute::inv(y.data<T>(), y.data<T>(), y.size());
      }
    };

  }
}
