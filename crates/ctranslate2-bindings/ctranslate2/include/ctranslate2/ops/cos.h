#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Cos : public UnaryOp {
    public:
      void operator()(const StorageView& x, StorageView& y) const {
        PROFILE("Cos");
        compute<float>(x, y);
      }

    private:
      template <typename T>
      void compute(const StorageView& x, StorageView& y) const {
        y.resize_as(x);
        primitives<>::cos(x.data<T>(), y.data<T>(), x.size());
      }
    };

  }
}
