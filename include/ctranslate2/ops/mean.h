#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Mean : public UnaryOp {
    public:
      Mean(const dim_t axis);

      void operator()(const StorageView& input, StorageView& output) const override;

    private:
      template <Device D, typename T>
      void compute(const StorageView& input,
                   const dim_t outer_size,
                   const dim_t axis_size,
                   const dim_t inner_size,
                   StorageView& output) const;

      const dim_t _axis;
    };

  }
}
