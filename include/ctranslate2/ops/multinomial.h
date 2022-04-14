#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    // Implements multinomial sampling with replacement.
    class Multinomial : public UnaryOp {
    public:
      Multinomial(dim_t sample_size = 1);
      void operator()(const StorageView& input, StorageView& output) const override;

    private:
      dim_t _sample_size;

      void dispatch(const StorageView& input, StorageView& output) const;

      template <Device D, typename T>
      void compute(const StorageView& input, StorageView& output) const;
    };

  }
}
