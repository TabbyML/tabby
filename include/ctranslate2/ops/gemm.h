#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Gemm : public Op {
    public:
      Gemm(float alpha = 1,
           float beta = 1,
           bool trans_a = false,
           bool trans_b = false,
           bool a_is_packed = false,
           bool b_is_packed = false);

      void operator()(const StorageView& a,
                      const StorageView& b,
                      StorageView& c,
                      const StorageView* a_shift_compensation = nullptr) const;
      void operator()(const StorageView& a,
                      const StorageView& b,
                      const StorageView& c,
                      StorageView& y) const;

    private:
      void compute(const StorageView& a,
                   const StorageView& b,
                   const StorageView* c,
                   StorageView& y,
                   const StorageView* a_shift_compensation) const;

      float _alpha;
      float _beta;
      bool _trans_a;
      bool _trans_b;
      bool _a_is_packed;
      bool _b_is_packed;
    };

  }
}
