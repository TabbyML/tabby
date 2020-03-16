#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class TopK : public Op {
    public:
      TopK(dim_t k, dim_t axis = -1);
      void operator()(const StorageView& x, StorageView& values, StorageView& indices) const;

    private:
      dim_t _k;

      template <Device D, typename DataType, typename IndexType>
      void compute(const StorageView& x,
                   StorageView& values,
                   StorageView& indices) const;

    };

  }
}
