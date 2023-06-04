#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Identity : public UnaryOp {
    public:
      void operator()(const StorageView& x, StorageView& y) const override {
        PROFILE("Identity");
        y = x;
      }
    };

  }
}
