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

      // Return the packed representation of b, if implemented by the GEMM backend.
      static StorageView pack_b_input(const StorageView& b,
                                      const bool transpose,
                                      const dim_t k,
                                      const dim_t n,
                                      const float alpha);

      // Return the compensation term when s8s8s32 is implemented with u8s8s32.
      static StorageView compensate_u8_input(const StorageView& b,
                                             const bool transpose,
                                             const dim_t k,
                                             const dim_t n,
                                             const float alpha);

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
