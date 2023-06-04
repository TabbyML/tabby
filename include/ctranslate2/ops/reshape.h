#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Reshape : public BinaryOp {
    public:
      void operator()(StorageView& data, const StorageView& shape) const {
        PROFILE("Reshape");
        data.reshape({shape.data<int32_t>(), shape.data<int32_t>() + shape.size()});
      }

      void operator()(const StorageView& data,
                      const StorageView& shape,
                      StorageView& reshaped) const override {
        PROFILE("Reshape");
        reshaped = data;
        reshaped.reshape({shape.data<int32_t>(), shape.data<int32_t>() + shape.size()});
      }
    };

  }
}
