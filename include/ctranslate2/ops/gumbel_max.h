#pragma once

#include "topk.h"

namespace ctranslate2 {
  namespace ops {

    class GumbelMax : public Op {
    public:
      GumbelMax(dim_t num_samples);

      void operator()(const StorageView& x, StorageView& values, StorageView& indices) const;
      void operator()(const StorageView& x, StorageView& indices) const;

    private:
      const dim_t _num_samples;
      const TopK _topk_op;

      template <Device D, typename T>
      void add_gumbel_noise(const StorageView& x, StorageView& y) const;
    };

  }
}
