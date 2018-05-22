#pragma once

#include "routines.h"
#include "storage_view.h"

namespace onmt {
  namespace ops {

    class LayerNorm {
    public:
      LayerNorm(const StorageView<float>& beta, const StorageView<float>& gamma)
        : _beta(beta)
        , _gamma(gamma) {
      }

      void operator()(const StorageView<float>& input, StorageView<float>& output) const {
        assert(input.rank() == 2);
        size_t batch_size = input.dim(0);
        size_t depth = input.dim(1);
        StorageView<float> tmp({depth});
        output.resize_as(input);
        for (size_t i = 0; i < batch_size; ++i) {
          const float* x = input.index({i});
          float* y = output.index({i});
          float mean = array_mean(x, depth);
          array_copy(x, y, depth);
          array_sub(mean, y, depth); // y is now centered
          array_pow(y, tmp.data(), 2, depth);
          float variance = array_mean(tmp.data(), depth);
          array_mul(1.0 / sqrt(variance + EPSILON), y, depth); // y is now centered and normalized.
          array_mul(_gamma.data(), y, depth);
          array_add(_beta.data(), y, depth);
        }
      }

    private:
      const StorageView<float>& _beta;
      const StorageView<float>& _gamma;
    };

    template <typename In, typename Out>
    class Gemm {
    public:
      Gemm(In alpha, Out beta, bool broadcast_c, bool trans_a, bool trans_b)
        : _alpha(alpha)
        , _beta(beta)
        , _broadcast_c(broadcast_c)
        , _trans_a(trans_a)
        , _trans_b(trans_b) {
      }

      void operator()(const StorageView<In>& a,
                      const StorageView<In>& b,
                      const StorageView<Out>* c,
                      StorageView<Out>& y) const {
        size_t k = a.dim(_trans_a ? -2 : -1);
        size_t n = b.dim(_trans_b ? -2 : -1);
        size_t m = a.size() / k; // Collapse leading dimensions.

        assert(k == b.dim(_trans_b ? -1 : -2));

        y.resize({m, n});

        if (_beta != static_cast<Out>(0)) {
          assert(c != nullptr);
          if (_broadcast_c) {
            assert(c->size() == n);
            for (size_t i = 0; i < m; ++i)
              array_copy(c->data(), y.index({i}), n);
          } else {
            assert(c->size() == y.size());
            array_copy(c->data(), y.data(), y.size());
          }
        }

        CBLAS_TRANSPOSE trans_a = _trans_a ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE trans_b = _trans_b ? CblasTrans : CblasNoTrans;

        sgemm(a.data(), b.data(),
              trans_a, trans_b,
              m, n, k,
              _alpha, _beta, y.data());

        // Restore collapsed dimensions.
        Shape output_shape(a.shape());
        output_shape.back() = n;
        y.reshape(output_shape);
      }

    private:
      In _alpha;
      Out _beta;
      bool _broadcast_c;
      bool _trans_a;
      bool _trans_b;
    };

    class MatMul {
    public:
      template <typename In, typename Out>
      void operator()(const StorageView<In>& a,
                      const StorageView<In>& b,
                      StorageView<Out>& y) const {
        operator()(a, b, false, false, y);
      }

      template <typename In, typename Out>
      void operator()(const StorageView<In>& a,
                      const StorageView<In>& b,
                      bool transpose_a,
                      bool transpose_b,
                      StorageView<Out>& y) const {
        size_t m, n, k;
        CBLAS_TRANSPOSE trans_a, trans_b;

        if (transpose_a) {
          m = a.dim(-1);
          k = a.dim(-2);
          trans_a = CblasTrans;
        } else {
          m = a.dim(-2);
          k = a.dim(-1);
          trans_a = CblasNoTrans;
        }

        if (transpose_b) {
          n = b.dim(-2);
          assert(k == b.dim(-1));
          trans_b = CblasTrans;
        } else {
          n = b.dim(-1);
          assert(k == b.dim(-2));
          trans_b = CblasNoTrans;
        }

        if (m * k != a.size()) {
          size_t batch_size = a.size() / (m * k);
          Shape output_shape(a.shape());
          output_shape[output_shape.size() - 1] = n;
          output_shape[output_shape.size() - 2] = m;
          y.resize(output_shape);
          batch_mat_mul(a.data(), b.data(), trans_a, trans_b, batch_size, m, n, k, y.data());
        } else {
          y.resize({m, n});
          mat_mul(a.data(), b.data(), trans_a, trans_b, m, n, k, y.data());
        }
      }
    };

    class ReLU {
    public:
      template <typename T>
      void operator()(StorageView<T>& x) const {
        for (size_t i = 0; i < x.size(); ++i) {
          if (x[i] < static_cast<T>(0))
            x[i] = static_cast<T>(0);
        }
      }

      template <typename T>
      void operator()(const StorageView<T>& input, StorageView<T>& output) const {
        output = input;
        operator()(output);
      }
    };

    class SoftMax {
    public:
      template <typename In, typename Out>
      void operator()(const StorageView<In>& input, StorageView<Out>& output) const {
        size_t depth = input.dim(-1);
        size_t batch_size = input.size() / depth;
        output.resize_as(input);
        for (size_t i = 0; i < batch_size; ++i) {
          const In* x = input.data() + (i * depth);
          Out* y = output.data() + (i * depth);
          In max = array_max(x, depth);
          array_copy(x, y, depth);
          array_sub(max, y, depth);
          array_exp(y, y, depth);
          Out sum = array_sum(y, depth);
          array_mul(1.f / (sum + EPSILON), y, depth);
        }
      }
    };

    template <typename T>
    class Gather {
    public:
      Gather(const StorageView<T>& from)
        : _from(from) {
        assert(from.rank() == 2);
      }

      void operator()(const StorageView<size_t>& input, StorageView<T>& output) const {
        size_t batch_size = input.dim(0);
        size_t depth = _from.dim(-1);
        output.resize({batch_size, depth});
        for (size_t i = 0; i < batch_size; ++i) {
          const T* src = _from.index({input[i]});
          T* dst = output.index({i});
          array_copy(src, dst, depth);
        }
      }

      size_t output_depth() const {
        return _from.dim(-1);
      }

    private:
      const StorageView<T>& _from;
    };

  }
}
