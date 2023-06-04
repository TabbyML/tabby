#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Sin : public UnaryOp {
    public:
      void operator()(const StorageView& x, StorageView& y) const {
        PROFILE("Sin");
        compute<float>(x, y);
      }

    private:
      template <typename T>
      void compute(const StorageView& x, StorageView& y) const {
        y.resize_as(x);
        primitives<>::sin(x.data<T>(), y.data<T>(), x.size());
      }
    };

  }
}
