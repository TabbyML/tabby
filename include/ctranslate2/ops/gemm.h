#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Gemm : public TernaryOp {
    public:
      Gemm(float alpha, float beta, bool broadcast_c, bool trans_a, bool trans_b);
      void operator()(const StorageView& a,
                      const StorageView& b,
                      const StorageView& c,
                      StorageView& y) const;

    private:
      template <Device D, typename In, typename Out = In>
      void compute(const StorageView& a,
                   const StorageView& b,
                   const StorageView& c,
                   StorageView& y) const {
        size_t k = a.dim(_trans_a ? -2 : -1);
        size_t n = b.dim(_trans_b ? -2 : -1);
        size_t m = a.size() / k; // Collapse leading dimensions.

        assert(k == b.dim(_trans_b ? -1 : -2));

        Shape output_shape(a.shape());
        output_shape[output_shape.size() - 1] = n;
        y.resize(output_shape);

        if (_beta != 0.f) {
          assert(!c.empty());
          if (_broadcast_c) {
            assert(c.size() == n);
            for (size_t i = 0; i < m; ++i)
              primitives<D>::copy(c.data<Out>(), y.data<Out>() + i * n, n);
          } else {
            assert(c.size() == y.size());
            primitives<D>::copy(c.data<Out>(), y.data<Out>(), y.size());
          }
        }

        primitives<D>::gemm(a.data<In>(), b.data<In>(),
                            _trans_a, _trans_b,
                            m, n, k,
                            _alpha, _beta,
                            y.data<Out>());
      }

      float _alpha;
      float _beta;
      bool _broadcast_c;
      bool _trans_a;
      bool _trans_b;
    };

  }
}
