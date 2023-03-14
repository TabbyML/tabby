#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class MedianFilter : public Op {
    public:
      MedianFilter(const dim_t width);
      void operator()(const StorageView& input, StorageView& output) const;

    private:
      const dim_t _width;
    };

  }
}
