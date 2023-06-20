#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Sin : public UnaryOp {
    public:
      void operator()(const StorageView& x, StorageView& y) const;
    };

  }
}
