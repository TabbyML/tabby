#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Dequantize : public Op {
    public:
      void operator()(const StorageView& input,
                      const StorageView& scale,
                      StorageView& output) const;

      // Rescales the int32 GEMM output to float32, given the input scales.
      void operator()(const StorageView& c,
                      const StorageView& a_scale,
                      const StorageView& b_scale,
                      const bool transpose_a,
                      const bool transpose_b,
                      StorageView& y) const;

    private:
      template <Device D, typename T>
      void dequantize(const StorageView& input,
                      const StorageView& scale,
                      StorageView& output) const;

      template <Device D>
      void dequantize_gemm_output(const StorageView& c,
                                  const StorageView& a_scale,
                                  const StorageView& b_scale,
                                  const bool transpose_a,
                                  const bool transpose_b,
                                  StorageView& y) const;

    };

  }
}
