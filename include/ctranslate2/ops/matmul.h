#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class MatMul : public BinaryOp {
    public:
      MatMul(bool trans_a = false, bool trans_b = false, float alpha = 1);
      void operator()(const StorageView& a,
                      const StorageView& b,
                      StorageView& y) const;

    private:
      bool _trans_a;
      bool _trans_b;
      float _alpha;

      template <Device D, typename In, typename Out = In>
      void compute(const StorageView& a,
                   const StorageView& b,
                   StorageView& y) const {
        dim_t m, k_a;
        if (_trans_a) {
          m = a.dim(-1);
          k_a = a.dim(-2);
        } else {
          m = a.dim(-2);
          k_a = a.dim(-1);
        }

        dim_t k_b, n;
        if (_trans_b) {
          n = b.dim(-2);
          k_b = b.dim(-1);
        } else {
          n = b.dim(-1);
          k_b = b.dim(-2);
        }

        if (k_a != k_b)
          throw std::invalid_argument("MatMul: k dimension of inputs a and b should match");

        const dim_t k = k_a;
        const dim_t a_batch_size = a.size() / (m * k);
        const dim_t b_batch_size = b.size() / (k * n);

        if (a_batch_size != b_batch_size)
          throw std::invalid_argument("MatMul: batch dimension of inputs a and b should match");

        const dim_t batch_size = a_batch_size;
        const float beta = 0;

        if (batch_size > 1) {
          Shape output_shape(a.shape());
          output_shape[output_shape.size() - 1] = n;
          output_shape[output_shape.size() - 2] = m;
          y.resize(std::move(output_shape));
          primitives<D>::gemm_batch(a.data<In>(), b.data<In>(),
                                    _trans_a, _trans_b,
                                    batch_size, m, n, k,
                                    _alpha, beta, y.data<Out>());
        } else {
          y.resize({m, n});
          primitives<D>::gemm(a.data<In>(), b.data<In>(),
                              /*a_is_packed=*/false, /*b_is_packed=*/false,
                              _trans_a, _trans_b,
                              m, n, k,
                              _alpha, beta, y.data<Out>());
        }
      }
    };

  }
}
