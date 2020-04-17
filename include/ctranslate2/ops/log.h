#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Log : public UnaryOp {
    public:
      void operator()(const StorageView& x, StorageView& y) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& x, StorageView& y) const {
        y.resize_as(x);
        primitives<D>::log(x.data<T>(), y.data<T>(), x.size());
      }
    };

  }
}
