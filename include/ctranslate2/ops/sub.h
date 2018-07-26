#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Sub : public BinaryOp {
    public:
      void operator()(const StorageView& a, const StorageView& b, StorageView& c) const override {
        DEVICE_DISPATCH(a.device(), TYPE_DISPATCH(a.dtype(), (compute<D, T>(a, b, c))));
      }

    private:
      template <Device D, typename T>
      void compute(const StorageView& a, const StorageView& b, StorageView& c) const {
        if (b.is_scalar()) {
          c = a;
          primitives<D>::sub(b.data<T>()[0], c.data<T>(), c.size());
        } else {
          c.resize_as(a);
          primitives<D>::sub(a.data<T>(), b.data<T>(), c.data<T>(), c.size());
        }
      }
    };

  }
}
