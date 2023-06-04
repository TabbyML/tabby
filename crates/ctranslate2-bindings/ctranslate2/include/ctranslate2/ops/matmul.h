#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class MatMul : public BinaryOp {
    public:
      MatMul(bool trans_a = false, bool trans_b = false, float alpha = 1);
      void operator()(const StorageView& a, const StorageView& b, StorageView& c) const;

    private:
      bool _trans_a;
      bool _trans_b;
      float _alpha;

      template <Device D, typename T>
      void compute(const StorageView& a, const StorageView& b, StorageView& c) const;
    };

  }
}
