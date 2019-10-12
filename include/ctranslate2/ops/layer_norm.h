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
                      StorageView& output) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& beta,
                   const StorageView& gamma,
                   const StorageView& input,
                   StorageView& output) const;
    };

  }
}
