#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Tanh : public UnaryOp {
    public:
      void operator()(const StorageView& x, StorageView& y) const override {
        PROFILE("Tanh");
        compute<float>(x, y);
      }

    private:
      template <typename T>
      void compute(const StorageView& x, StorageView& y) const {
        y.resize_as(x);
        primitives<>::tanh(x.data<T>(), y.data<T>(), x.size());
      }
    };

  }
}
