#pragma once

#include "opennmt/ops/op.h"

namespace opennmt {
  namespace ops {

    class MatMul : public BinaryOp {
    public:
      MatMul()
        : _trans_a(false)
        , _trans_b(false) {
      }
      MatMul(bool trans_a, bool trans_b)
        : _trans_a(trans_a)
        , _trans_b(trans_b) {
      }

      void operator()(const StorageView& a,
                      const StorageView& b,
                      StorageView& y) const {
        switch (a.dtype()) {
        case DataType::DT_INT16:
          return compute<int16_t, int32_t>(a, b, y);
        case DataType::DT_FLOAT:
          return compute<float>(a, b, y);
        default:
          throw std::invalid_argument("unsupported compute type " + dtype_name(a.dtype()));
        }
      }

    private:
      bool _trans_a;
      bool _trans_b;

      template <typename In, typename Out = In>
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

        In alpha = 1;
        Out beta = 0;

        if (m * k != a.size()) {
          size_t batch_size = a.size() / (m * k);
          Shape output_shape(a.shape());
          output_shape[output_shape.size() - 1] = n;
          output_shape[output_shape.size() - 2] = m;
          y.resize(output_shape);
          primitives::gemm_batch(a.data<In>(), b.data<In>(),
                                 _trans_a, _trans_b,
                                 batch_size, m, n, k,
                                 alpha, beta, y.data<Out>());
        } else {
          y.resize({m, n});
          primitives::gemm(a.data<In>(), b.data<In>(),
                           _trans_a, _trans_b,
                           m, n, k,
                           alpha, beta, y.data<Out>());
        }
      }
    };

  }
}
