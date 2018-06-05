#pragma once

#include "opennmt/ops/op.h"

namespace opennmt {
  namespace ops {

    class Add : public BinaryOp {
    public:
      void operator()(const StorageView& a, const StorageView& b, StorageView& c) const override {
        TYPE_DISPATCH(a.dtype(), compute<T>(a, b, c));
      }

    private:
      template <typename T>
      void compute(const StorageView& a, const StorageView& b, StorageView& c) const {
        if (b.is_scalar()) {
          c = a;
          compute::add(b.data<T>()[0], c.data<T>(), c.size());
        } else {
          c.resize_as(a);
          compute::add(a.data<T>(), b.data<T>(), c.data<T>(), c.size());
        }
      }
    };

  }
}
