#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Gather : public BinaryOp {
    public:
      Gather(const dim_t axis = 0, const dim_t batch_dims = 0);
      using BinaryOp::operator();

      void operator()(StorageView& data, const StorageView& input) const;
      void operator()(const StorageView& data,
                      const StorageView& input,
                      StorageView& output) const override;

    private:
      template <Device D, typename T>
      void compute(const StorageView& data,
                   const StorageView& input,
                   const dim_t axis,
                   const dim_t batch_dims,
                   StorageView& output) const;

      const dim_t _axis;
      const dim_t _batch_dims;
    };

  }
}
