#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class LogSoftMax : public UnaryOp {
    public:
      using UnaryOp::operator();
      void operator()(const StorageView& x, StorageView& y) const override {
        operator()(x, nullptr, y);
      }
      void operator()(const StorageView& x, const StorageView& lengths, StorageView& y) const {
        operator()(x, &lengths, y);
      }
      void operator()(const StorageView& x, const StorageView* lengths, StorageView& y) const {
        if (lengths && lengths->dim(0) == 1)  // Disable masking when batch size is 1.
          lengths = nullptr;
        y.resize_as(x);
        DEVICE_DISPATCH(x.device(), (compute<D, float>(x, lengths, y)));
      }

    private:
      template <Device D, typename T>
      void compute(const StorageView& input, const StorageView* lengths, StorageView& output) const;
    };

  }
}
