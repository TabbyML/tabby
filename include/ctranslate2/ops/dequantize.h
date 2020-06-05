#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Dequantize : public BinaryOp {
    public:
      void operator()(const StorageView& x, const StorageView& scale, StorageView& y) const override;

      // Rescales the int32 GEMM output to float32, given the input scales.
      void operator()(const StorageView& c,
                      const StorageView& a_scale,
                      const StorageView& b_scale,
                      const bool transpose_a,
                      const bool transpose_b,
                      StorageView& y) const;

    };

  }
}
