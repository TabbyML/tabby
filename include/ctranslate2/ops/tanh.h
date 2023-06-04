#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Tanh : public UnaryOp {
    public:
      void operator()(const StorageView& x, StorageView& y) const;
    };

  }
}
