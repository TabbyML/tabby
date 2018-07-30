#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class SoftMax : public UnaryOp {
    public:
      using UnaryOp::operator();
      void operator()(const StorageView& x, StorageView& y) const override {
        DEVICE_DISPATCH(x.device(), (compute<D, float>(x, y)));
      }

    private:
      template <Device D, typename T>
      void compute(const StorageView& input, StorageView& output) const;
    };

  }
}
