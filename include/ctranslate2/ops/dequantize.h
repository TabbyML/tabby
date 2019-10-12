#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Dequantize : public BinaryOp {
    public:
      void operator()(const StorageView& x, const StorageView& scale, StorageView& y) const override;
      void operator()(const StorageView& gemm_output,
                      const StorageView& input_scale,
                      const StorageView& weight_scale,
                      StorageView& output) const;

    };

  }
}
