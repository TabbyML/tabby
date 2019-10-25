#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Sigmoid : public UnaryOp {
    public:
      void operator()(const StorageView& x, StorageView& y) const override {
        PROFILE_FUN;
        compute<float>(x, y);
      }

    private:
      template <typename T>
      void compute(const StorageView& x, StorageView& y) const {
        y = x;
        primitives<>::mul(static_cast<T>(-1), y.data<T>(), y.size());
        primitives<>::exp(y.data<T>(), y.data<T>(), y.size());
        primitives<>::add(static_cast<T>(1), y.data<T>(), y.size());
        primitives<>::inv(y.data<T>(), y.data<T>(), y.size());
      }
    };

  }
}
