#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class MatMul : public BinaryOp {
    public:
      MatMul(bool trans_a = false, bool trans_b = false, float alpha = 1)
        : _trans_a(trans_a)
        , _trans_b(trans_b)
        , _alpha(alpha) {
      }

      void operator()(const StorageView& a,
                      const StorageView& b,
                      StorageView& y) const {
        switch (a.dtype()) {
        case DataType::DT_FLOAT:
          DEVICE_DISPATCH(a.device(), (compute<D, float>(a, b, y)));
          break;
        default:
          throw std::invalid_argument("unsupported compute type " + dtype_name(a.dtype()));
        }
      }

    private:
      bool _trans_a;
      bool _trans_b;
      float _alpha;

      template <Device D, typename In, typename Out = In>
      void compute(const StorageView& a,
                   const StorageView& b,
                   StorageView& y) const {
        size_t m, n, k;

        if (_trans_a) {
          m = a.dim(-1);
          k = a.dim(-2);
        } else {
          m = a.dim(-2);
          k = a.dim(-1);
        }

        if (_trans_b) {
          n = b.dim(-2);
          assert(k == b.dim(-1));
        } else {
          n = b.dim(-1);
          assert(k == b.dim(-2));
        }

        float beta = 0;

        if (m * k != a.size()) {
          size_t batch_size = a.size() / (m * k);
          Shape output_shape(a.shape());
          output_shape[output_shape.size() - 1] = n;
          output_shape[output_shape.size() - 2] = m;
          y.resize(output_shape);
          primitives<D>::gemm_batch(a.data<In>(), b.data<In>(),
                                    _trans_a, _trans_b,
                                    batch_size, m, n, k,
                                    _alpha, beta, y.data<Out>());
        } else {
          y.resize({m, n});
          primitives<D>::gemm(a.data<In>(), b.data<In>(),
                              _trans_a, _trans_b,
                              m, n, k,
                              _alpha, beta, y.data<Out>());
        }
      }
    };

  }
}
