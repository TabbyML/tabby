#pragma once

#include "activation.h"
#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Dequantize : public Op {
    public:
      Dequantize(const ActivationType* activation_type = nullptr);

      void operator()(const StorageView& input,
                      const StorageView& scale,
                      StorageView& output) const;

      // Rescales the int32 GEMM output to float32, given the input scales.
      void operator()(const StorageView& c,
                      const StorageView& a_scale,
                      const StorageView& b_scale,
                      const bool transpose_a,
                      const bool transpose_b,
                      StorageView& y,
                      const StorageView* bias = nullptr) const;

    private:
      template <Device D, typename InT, typename OutT>
      void dequantize(const StorageView& input,
                      const StorageView& scale,
                      StorageView& output) const;

      template <Device D, typename T>
      void dequantize_gemm_output(const StorageView& c,
                                  const StorageView& a_scale,
                                  const StorageView& b_scale,
                                  const bool transpose_a,
                                  const bool transpose_b,
                                  const StorageView* bias,
                                  StorageView& y) const;

      const ActivationType* _activation_type;
    };

  }
}
