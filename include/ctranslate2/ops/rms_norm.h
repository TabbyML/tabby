#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class RMSNorm : public Op {
    public:
      void operator()(const StorageView& gamma,
                      const StorageView& input,
                      StorageView& output) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& gamma,
                   const StorageView& input,
                   StorageView& output) const;

      const float _epsilon = 1e-6;
    };

  }
}
