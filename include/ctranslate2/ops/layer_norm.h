#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class LayerNorm : public TernaryOp {
    public:
      using TernaryOp::operator();
      void operator()(const StorageView& beta,
                      const StorageView& gamma,
                      const StorageView& input,
                      StorageView& output) const {
        DEVICE_DISPATCH(input.device(), (compute<D, float>(beta, gamma, input, output)));
      }

    private:
      template <Device D, typename T>
      void compute(const StorageView& beta,
                   const StorageView& gamma,
                   const StorageView& input,
                   StorageView& output) const;
    };

  }
}
