#pragma once

#include "compute.h"
#include "storage_view.h"

#define EPSILON 0.000001f

namespace onmt {
  namespace ops {

    class Concat {
    public:
      Concat(int axis)
        : _axis(axis) {
      }

      template <typename T>
      void operator()(const std::vector<StorageView<T>*>& inputs,
                      StorageView<T>& output) const {
        size_t rank = inputs.front()->rank();
        size_t axis = _axis < 0 ? rank + _axis : _axis;
        size_t concat_dims = 0;
        for (const auto& x : inputs) {
          concat_dims += x->dim(axis);
        }

        Shape output_shape(inputs.front()->shape());
        output_shape[axis] = concat_dims;
        output.resize(output_shape);

        size_t offset = 0;
        for (const auto& x : inputs) {
          size_t iter_dim = 1;
          size_t copy_dim = 1;
          for (size_t i = 0; i < axis; ++i)
            iter_dim *= x->dim(i);
          for (size_t i = axis; i < x->rank(); ++i)
            copy_dim *= x->dim(i);
          for (size_t i = 0; i < iter_dim; ++i) {
            compute::copy(x->data() + i * copy_dim,
                          output.data() + offset + i * concat_dims * output.stride(axis),
                          copy_dim);
          }
          offset += copy_dim;
        }
      }

    private:
      int _axis;
    };

    class Transpose {
    public:
      template <typename T>
      void operator()(StorageView<T>& x) const {
        size_t depth = x.dim(-1);
        size_t batch_size = x.size() / depth;
        compute::transpose_2d_inplace(x.data(), batch_size, depth);
      }

      template <typename T>
      void operator()(const StorageView<T>& x, StorageView<T>& y) const {
        size_t depth = x.dim(-1);
        size_t batch_size = x.size() / depth;
        compute::transpose_2d(x.data(), batch_size, depth, y.data());
      }
    };

    class Unsqueeze {
    public:
      Unsqueeze(const std::vector<size_t>& axes)
        : _axes(axes) {
        std::sort(_axes.begin(), _axes.end());
      }

      template <typename T>
      void operator()(StorageView<T>& data) const {
        Shape new_shape;
        for (size_t i = 0, j = 0; i < data.rank(); ++i) {
          if (i == _axes[j]) {
            ++j;
            new_shape.push_back(1);
          }
          new_shape.push_back(data.dim(i));
        }
        data.reshape(new_shape);
      }

    private:
      std::vector<size_t> _axes;
    };

    class LayerNorm {
    public:
      template <typename T>
      void operator()(const StorageView<T>& beta,
                      const StorageView<T>& gamma,
                      const StorageView<T>& input,
                      StorageView<T>& output) const {
        size_t depth = input.dim(-1);
        size_t batch_size = input.size() / depth;
        StorageView<float> tmp({depth});
        output.resize_as(input);
        for (size_t i = 0; i < batch_size; ++i) {
          const T* x = input.index({i});
          T* y = output.index({i});
          T mean = compute::mean(x, depth);
          compute::copy(x, y, depth);
          compute::sub(mean, y, depth);
          compute::pow(y, tmp.data(), static_cast<T>(2), depth);
          T variance = compute::mean(tmp.data(), depth);
          compute::mul(static_cast<T>(1.0 / sqrt(variance + EPSILON)), y, depth);
          compute::mul(gamma.data(), y, depth);
          compute::add(beta.data(), y, depth);
        }
      }
    };

    class Gemm {
    public:
      Gemm(float alpha, float beta, bool broadcast_c, bool trans_a, bool trans_b)
        : _alpha(alpha)
        , _beta(beta)
        , _broadcast_c(broadcast_c)
        , _trans_a(trans_a)
        , _trans_b(trans_b) {
      }

      template <typename In, typename Out>
      void operator()(const StorageView<In>& a,
                      const StorageView<In>& b,
                      const StorageView<Out>* c,
                      StorageView<Out>& y) const {
        size_t k = a.dim(_trans_a ? -2 : -1);
        size_t n = b.dim(_trans_b ? -2 : -1);
        size_t m = a.size() / k; // Collapse leading dimensions.

        assert(k == b.dim(_trans_b ? -1 : -2));

        Shape output_shape(a.shape());
        output_shape[output_shape.size() - 1] = n;
        output_shape[output_shape.size() - 2] = m;
        y.resize(output_shape);

        if (_beta != 0.f) {
          assert(c != nullptr);
          if (_broadcast_c) {
            assert(c->size() == n);
            for (size_t i = 0; i < m; ++i)
              compute::copy(c->data(), y.index({i}), n);
          } else {
            assert(c->size() == y.size());
            compute::copy(c->data(), y.data(), y.size());
          }
        }

        compute::gemm(a.data(), b.data(),
                      _trans_a, _trans_b,
                      m, n, k,
                      static_cast<In>(_alpha), static_cast<Out>(_beta),
                      y.data());
      }

    private:
      float _alpha;
      float _beta;
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

        if (transpose_a) {
          m = a.dim(-1);
          k = a.dim(-2);
        } else {
          m = a.dim(-2);
          k = a.dim(-1);
        }

        if (transpose_b) {
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
          compute::gemm_batch(a.data(), b.data(),
                              transpose_a, transpose_b,
                              batch_size, m, n, k,
                              alpha, beta, y.data());
        } else {
          y.resize({m, n});
          compute::gemm(a.data(), b.data(),
                        transpose_a, transpose_b,
                        m, n, k,
                        alpha, beta, y.data());
        }
      }
    };

    class Identity {
    public:
      template <typename T>
      void operator()(const StorageView<T>& x, StorageView<T>& y) const {
        y = x;
      }
    };

    class Cos {
    public:
      template <typename T>
      void operator()(const StorageView<T>& x, StorageView<T>& y) const {
        y.resize_as(x);
        compute::cos(x.data(), y.data(), x.size());
      }
    };

    class Sin {
    public:
      template <typename T>
      void operator()(const StorageView<T>& x, StorageView<T>& y) const {
        y.resize_as(x);
        compute::sin(x.data(), y.data(), x.size());
      }
    };

    class Add {
    public:
      template <typename T>
      void operator()(const StorageView<T>& a, T b, StorageView<T>& c) const {
        c = a;
        compute::add(b, c.data(), c.size());
      }

      template <typename T>
      void operator()(const StorageView<T>& a, const StorageView<T>& b, StorageView<T>& c) const {
        c = a;
        compute::add(b.data(), c.data(), c.size());
      }
    };

    class Mul {
    public:
      template <typename T>
      void operator()(const StorageView<T>& a, T b, StorageView<T>& c) const {
        c = a;
        compute::mul(b, c.data(), c.size());
      }

      template <typename T>
      void operator()(const StorageView<T>& a, const StorageView<T>& b, StorageView<T>& c) const {
        c.resize_as(a);
        compute::mul(a.data(), b.data(), c.data(), a.size());
      }
    };

    class Reshape {
    public:
      template <typename T>
      void operator()(StorageView<T>& data, const std::vector<size_t>& shape) const {
        data.reshape(shape);
      }
      template <typename T>
      void operator()(const StorageView<T>& data, const std::vector<size_t>& shape, StorageView<T>& reshaped) const {
        reshaped = data;
        reshaped.reshape(shape);
      }
    };

    class ReLU {
    public:
      template <typename T>
      void operator()(StorageView<T>& x) const {
        compute::relu(x.data(), x.size());
      }

      template <typename T>
      void operator()(const StorageView<T>& x, StorageView<T>& y) const {
        y.resize_as(x);
        compute::relu(x.data(), y.data(), x.size());
      }
    };

    class Tanh {
    public:
      template <typename T>
      void operator()(const StorageView<T>& x, StorageView<T>& y) const {
        y.resize_as(x);
        compute::tanh(x.data(), y.data(), x.size());
      }
    };

    class Sigmoid {
    public:
      template <typename T>
      void operator()(const StorageView<T>& x, StorageView<T>& y) const {
        y = x;
        compute::mul(-1, y.data(), y.size());
        compute::exp(y.data(), y.data(), y.size());
        compute::add(1, y.data(), y.size());
        compute::inv(y.data(), y.data(), y.size());
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
          In max = compute::max(x, depth);
          compute::copy(x, y, depth);
          compute::sub(max, y, depth);
          compute::exp(y, y, depth);
          Out sum = compute::sum(y, depth);
          compute::mul(1.f / (sum + EPSILON), y, depth);
        }
      }
    };

    class Gather {
    public:
      Gather(int axis = 0)
        : _axis(axis) {
        if (axis != 0)
          throw std::invalid_argument("unsupported gather axis " + std::to_string(axis));
      }
      template <typename T>
      void operator()(const StorageView<T>& data,
                      const StorageView<size_t>& input,
                      StorageView<T>& output) const {
        size_t batch_size = input.dim(0);
        size_t depth = data.dim(-1);
        output.resize({batch_size, depth});
        for (size_t i = 0; i < batch_size; ++i) {
          const T* src = data.index({input[i]});
          T* dst = output.index({i});
          compute::copy(src, dst, depth);
        }
      }

    private:
      int _axis;
    };

    class Quantize {
    public:
      Quantize(float scale = 1, float shift = 0)
        : _scale(scale)
        , _shift(shift) {
      }

      template <typename In, typename Out>
      void operator()(const StorageView<In>& x, StorageView<Out>& y) const {
        y.resize_as(x);
        compute::quantize(x.data(), y.data(), x.size(), _scale, _shift);
      }

    private:
      float _scale;
      float _shift;
    };

    class Unquantize {
    public:
      Unquantize(float scale = 1, float shift = 0)
        : _scale(scale)
        , _shift(shift) {
      }

      template <typename In, typename Out>
      void operator()(const StorageView<In>& x, StorageView<Out>& y) const {
        y.resize_as(x);
        compute::unquantize(x.data(), y.data(), x.size(), _scale, _shift);
      }

    private:
      float _scale;
      float _shift;
    };

  }
}
