#pragma once

#include "activation.h"
#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Conv1D : public Op {
    public:
      Conv1D(dim_t stride = 1, dim_t padding = 0, dim_t dilation = 1);

      void operator()(const StorageView& input,
                      const StorageView& weight,
                      const StorageView& bias,
                      StorageView& output) const;

      void operator()(const StorageView& input,
                      const StorageView& weight,
                      StorageView& output) const;

    private:
      dim_t _stride;
      dim_t _padding;
      dim_t _dilation;

      void operator()(const StorageView& input,
                      const StorageView& weight,
                      const StorageView* bias,
                      StorageView& output) const;

      template <Device D, typename T>
      void compute(const StorageView& input,
                   const StorageView& weight,
                   const StorageView* bias,
                   StorageView& output) const;
    };

  }
}
