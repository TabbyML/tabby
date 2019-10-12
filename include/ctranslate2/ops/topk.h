#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class TopK : public Op {
    public:
      TopK(size_t k, int axis = -1);
      void operator()(const std::vector<StorageView*>& inputs,
                      std::vector<StorageView*>& outputs) const override;
      void operator()(const StorageView& x, StorageView& values, StorageView& indices) const;

    private:
      size_t _k;

      template <Device D, typename DataType, typename IndexType>
      void compute(const StorageView& x,
                   StorageView& values,
                   StorageView& indices) const;

    };

  }
}
