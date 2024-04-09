#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Rotary : public Op {
    public:
      Rotary(const dim_t ndims, const bool interleave);

      void operator()(const StorageView& input,
                      const StorageView& sin,
                      const StorageView& cos,
                      StorageView& output,
                      bool is_transpose=true) const;

    private:
      const dim_t _ndims;
      const bool _interleave;

      template <Device D, typename T>
      void compute(const StorageView& input,
                   const StorageView& sin,
                   const StorageView& cos,
                   StorageView& output,
                   bool is_transpose) const;
    };

  }
}
