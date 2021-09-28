#pragma once

#include "activation.h"
#include "op.h"

namespace ctranslate2 {
  namespace ops {

    void apply_bias_and_activation(StorageView& x,
                                   const StorageView* bias,
                                   const ActivationType* activation_type);

    class Gemm : public Op {
    public:
      Gemm(float alpha = 1,
           float beta = 1,
           bool trans_a = false,
           bool trans_b = false,
           bool a_is_packed = false,
           bool b_is_packed = false,
           const ActivationType* activation_type = nullptr);

      void operator()(const StorageView& a,
                      const StorageView& b,
                      StorageView& c,
                      const StorageView* a_shift_compensation = nullptr,
                      const StorageView* bias = nullptr) const;

    private:
      float _alpha;
      float _beta;
      bool _trans_a;
      bool _trans_b;
      bool _a_is_packed;
      bool _b_is_packed;
      const ActivationType* _activation_type;

      template <Device D, typename In, typename Out>
      void compute(const StorageView& a,
                   const StorageView& b,
                   StorageView& c,
                   const StorageView* a_shift_compensation) const;
    };

  }
}
