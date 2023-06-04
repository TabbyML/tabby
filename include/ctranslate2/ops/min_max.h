#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Min : public BinaryOp {
    public:
      void operator()(const StorageView& a, const StorageView& b, StorageView& c) const override;

    private:
      template <Device D, typename T>
      void compute(const StorageView& a, const StorageView& b, StorageView& c) const {
        c.resize_as(a);
        if (b.is_scalar()) {
          primitives<D>::min(b.data<T>()[0], a.data<T>(), c.data<T>(), c.size());
        } else {
          primitives<D>::min(a.data<T>(), b.data<T>(), c.data<T>(), c.size());
        }
      }
    };

    class Max : public BinaryOp {
    public:
      void operator()(const StorageView& a, const StorageView& b, StorageView& c) const override;

    private:
      template <Device D, typename T>
      void compute(const StorageView& a, const StorageView& b, StorageView& c) const {
        c.resize_as(a);
        if (b.is_scalar()) {
          primitives<D>::max(b.data<T>()[0], a.data<T>(), c.data<T>(), c.size());
        } else {
          primitives<D>::max(a.data<T>(), b.data<T>(), c.data<T>(), c.size());
        }
      }
    };

  }
}
