#pragma once

#include "compute.h"
#include "storage_view.h"

#define EPSILON 0.000001f

namespace onmt {
  namespace ops {

    class Op {
    public:
      virtual ~Op() = default;
      virtual void operator()(const std::vector<StorageView*>& inputs,
                              std::vector<StorageView*>& outputs) const = 0;
    };

    class UnaryOp : public Op {
    public:
      virtual void operator()(const std::vector<StorageView*>& inputs,
                              std::vector<StorageView*>& outputs) const override {
        operator()(*inputs[0], *outputs[0]);
      }
      virtual void operator()(const StorageView&, StorageView&) const = 0;
    };
    class BinaryOp : public Op {
    public:
      virtual void operator()(const std::vector<StorageView*>& inputs,
                              std::vector<StorageView*>& outputs) const override {
        operator()(*inputs[0], *inputs[1], *outputs[0]);
      }
      virtual void operator()(const StorageView&, const StorageView&, StorageView&) const = 0;
    };
    class TernaryOp : public Op {
    public:
      virtual void operator()(const std::vector<StorageView*>& inputs,
                              std::vector<StorageView*>& outputs) const override {
        operator()(*inputs[0], *inputs[1], *inputs[2], *outputs[0]);
      }
      virtual void operator()(const StorageView&,
                              const StorageView&,
                              const StorageView&,
                              StorageView&) const = 0;
    };


    class Concat : public Op {
    public:
      Concat(int axis)
        : _axis(axis) {
      }

      void operator()(const std::vector<StorageView*>& inputs,
                      std::vector<StorageView*>& outputs) const override {
        operator()(inputs, *outputs[0]);
      }
      void operator()(const std::vector<StorageView*>& inputs,
                      StorageView& output) const {
        TYPE_DISPATCH(output.dtype(), compute<T>(inputs, output));
      }

    private:
      int _axis;

      template <typename T>
      void compute(const std::vector<StorageView*>& inputs,
                   StorageView& output) const {
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
            compute::copy(x->data<T>() + i * copy_dim,
                          output.data<T>() + offset + i * concat_dims * output.stride(axis),
                          copy_dim);
          }
          offset += copy_dim;
        }
      }
    };

    class Split : public Op {
    public:
      Split(int axis)
        : _axis(axis) {
      }
      Split(int axis, const std::vector<int>& split)
        : _axis(axis)
        , _split(split) {
      }

      void operator()(const std::vector<StorageView*>& inputs,
                      std::vector<StorageView*>& outputs) const override {
        operator()(*inputs[0], outputs);
      }
      void operator()(const StorageView& input,
                      std::vector<StorageView*>& outputs) const {
        TYPE_DISPATCH(input.dtype(), compute<T>(input, outputs));
      }

    private:
      int _axis;
      std::vector<int> _split;

      template <typename T>
      void compute(const StorageView& input,
                   std::vector<StorageView*>& outputs) const {
        size_t rank = input.rank();
        size_t axis = _axis < 0 ? rank + _axis : _axis;
        size_t offset = 0;
        if (!_split.empty())
          assert(_split.size() == outputs.size());
        for (size_t j = 0; j < outputs.size(); ++j) {
          auto& x = *outputs[j];
          auto shape = input.shape();
          shape[axis] = _split.empty() ? input.dim(axis) / outputs.size() : _split[j];
          x.resize(shape);
          size_t iter_dim = 1;
          size_t copy_dim = 1;
          for (size_t i = 0; i < axis; ++i)
            iter_dim *= x.dim(i);
          for (size_t i = axis; i < x.rank(); ++i)
            copy_dim *= x.dim(i);
          for (size_t i = 0; i < iter_dim; ++i) {
            compute::copy(input.data<T>() + offset + i * input.dim(axis) * input.stride(axis),
                          x.data<T>() + i * copy_dim,
                          copy_dim);
          }
          offset += copy_dim;
        }
      }

    };

    class Transpose : public UnaryOp {
    public:
      Transpose() = default;
      Transpose(const std::vector<size_t>& perm)
        : _perm(perm) {
      }

      void operator()(const StorageView& x, StorageView& y) const override {
        if (x.rank() == 1) {
          y = x;
          return;
        }

        std::vector<size_t> perm;
        bool identity = true;
        if (_perm.empty()) {
          perm.resize(x.rank());
          for (size_t i = 0; i < x.rank(); ++i)
            perm[i] = x.rank() - i - 1;
          identity = false;
        } else {
          assert(_perm.size() == x.rank());
          perm = _perm;
          for (size_t i = 0; i < x.rank(); ++i) {
            if (perm[i] != i) {
              identity = false;
              break;
            }
          }
        }

        if (identity) {
          y = x;
          return;
        }

        TYPE_DISPATCH(x.dtype(), compute<T>(x, perm, y));
      }

    private:
      std::vector<size_t> _perm;

      template <typename T>
      void compute(const StorageView& x, const std::vector<size_t>& perm, StorageView& y) const {
        if (x.rank() == 2) {
          y.resize({x.dim(1), x.dim(0)});
          compute::transpose_2d(x.data<T>(), x.shape().data(), y.data<T>());
        } else if (x.rank() == 3) {
          y.resize({x.dim(perm[0]), x.dim(perm[1]), x.dim(perm[2])});
          compute::transpose_3d(x.data<T>(), x.shape().data(), perm.data(), y.data<T>());
        } else if (x.rank() == 4) {
          y.resize({x.dim(perm[0]), x.dim(perm[1]), x.dim(perm[2]), x.dim(perm[3])});
          compute::transpose_4d(x.data<T>(), x.shape().data(), perm.data(), y.data<T>());
        } else {
          throw std::invalid_argument("unsupported rank for transposition");
        }
      }
    };

    class Unsqueeze : public UnaryOp {
    public:
      Unsqueeze(const std::vector<size_t>& axes)
        : _axes(axes) {
        std::sort(_axes.begin(), _axes.end());
      }

      void operator()(StorageView& data) const {
        data.reshape(transform_shape(data.shape()));
      }
      void operator()(const StorageView& data, StorageView& expanded) const override {
        expanded.shallow_copy(const_cast<StorageView&>(data));
        expanded.reshape(transform_shape(data.shape()));
      }

    private:
      std::vector<size_t> _axes;

      Shape transform_shape(const Shape& shape) const {
        Shape new_shape;
        for (size_t i = 0, j = 0; i < shape.size(); ++i) {
          if (j < _axes.size() && i == _axes[j]) {
            ++j;
            new_shape.push_back(1);
          }
          new_shape.push_back(shape[i]);
        }
        return new_shape;
      }
    };

    class Squeeze : public UnaryOp {
    public:
      Squeeze(const std::vector<size_t>& axes)
        : _axes(axes) {
        std::sort(_axes.begin(), _axes.end());
      }

      void operator()(StorageView& data) const {
        data.reshape(transform_shape(data.shape()));
      }
      void operator()(const StorageView& data, StorageView& squeezed) const override {
        squeezed.shallow_copy(const_cast<StorageView&>(data));
        squeezed.reshape(transform_shape(data.shape()));
      }

    private:
      std::vector<size_t> _axes;

      Shape transform_shape(const Shape& shape) const {
        Shape new_shape;
        for (size_t i = 0, j = 0; i < shape.size(); ++i) {
          if (j < _axes.size() && i == _axes[j]) {
            if (shape[i] != 1)
              throw std::invalid_argument("can't squeeze dimension greater than 1");
            ++j;
          } else {
            new_shape.push_back(shape[i]);
          }
        }
        return new_shape;
      }

    };

    class Reshape {
    public:
      void operator()(StorageView& data, const StorageView& shape) const {
        data.reshape(std::vector<size_t>(shape.data<int32_t>(),
                                         shape.data<int32_t>() + shape.size()));
      }

      void operator()(const StorageView& data,
                      const StorageView& shape,
                      StorageView& reshaped) const {
        reshaped = data;
        reshaped.reshape(std::vector<size_t>(shape.data<int32_t>(),
                                             shape.data<int32_t>() + shape.size()));
      }
    };

    class LayerNorm : public TernaryOp {
    public:
      void operator()(const StorageView& beta,
                      const StorageView& gamma,
                      const StorageView& input,
                      StorageView& output) const {
        compute<float>(beta, gamma, input, output);
      }

    private:
      template <typename T>
      void compute(const StorageView& beta,
                   const StorageView& gamma,
                   const StorageView& input,
                   StorageView& output) const {
        size_t depth = input.dim(-1);
        size_t batch_size = input.size() / depth;
        StorageView tmp({depth}, input.dtype());
        output.resize_as(input);
        for (size_t i = 0; i < batch_size; ++i) {
          const auto* x = input.index<T>({i});
          auto* y = output.index<T>({i});
          auto mean = compute::mean(x, depth);
          compute::copy(x, y, depth);
          compute::sub(mean, y, depth);
          compute::pow(y, tmp.data<T>(), static_cast<T>(2), depth);
          auto variance = compute::mean(tmp.data<T>(), depth);
          compute::mul(static_cast<T>(1.0 / sqrt(variance + EPSILON)), y, depth);
          compute::mul(gamma.data<T>(), y, depth);
          compute::add(beta.data<T>(), y, depth);
        }
      }
    };

    class Gemm : public TernaryOp {
    public:
      Gemm(float alpha, float beta, bool broadcast_c, bool trans_a, bool trans_b)
        : _alpha(alpha)
        , _beta(beta)
        , _broadcast_c(broadcast_c)
        , _trans_a(trans_a)
        , _trans_b(trans_b) {
      }

      void operator()(const StorageView& a,
                      const StorageView& b,
                      const StorageView& c,
                      StorageView& y) const {
        switch (a.dtype()) {
        case DataType::DT_INT16:
          return compute<int16_t, int32_t>(a, b, c, y);
        case DataType::DT_FLOAT:
          return compute<float>(a, b, c, y);
        default:
          throw std::invalid_argument("unsupported compute type " + dtype_name(a.dtype()));
        }
      }

    private:
      template <typename In, typename Out = In>
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
        output_shape[output_shape.size() - 2] = m;
        y.resize(output_shape);

        if (_beta != 0.f) {
          assert(!c.empty());
          if (_broadcast_c) {
            assert(c.size() == n);
            for (size_t i = 0; i < m; ++i)
              compute::copy(c.data<Out>(), y.index<Out>({i}), n);
          } else {
            assert(c.size() == y.size());
            compute::copy(c.data<Out>(), y.data<Out>(), y.size());
          }
        }

        compute::gemm(a.data<In>(), b.data<In>(),
                      _trans_a, _trans_b,
                      m, n, k,
                      static_cast<In>(_alpha), static_cast<Out>(_beta),
                      y.data<Out>());
      }

      float _alpha;
      float _beta;
      bool _broadcast_c;
      bool _trans_a;
      bool _trans_b;
    };

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
          compute::gemm_batch(a.data<In>(), b.data<In>(),
                              _trans_a, _trans_b,
                              batch_size, m, n, k,
                              alpha, beta, y.data<Out>());
        } else {
          y.resize({m, n});
          compute::gemm(a.data<In>(), b.data<In>(),
                        _trans_a, _trans_b,
                        m, n, k,
                        alpha, beta, y.data<Out>());
        }
      }
    };


    class Identity : public UnaryOp {
    public:
      void operator()(const StorageView& x, StorageView& y) const override {
        y = x;
      }
    };

    class ReLU : public UnaryOp {
    public:
      void operator()(StorageView& x) const {
        TYPE_DISPATCH(x.dtype(), compute<T>(x));
      }
      void operator()(const StorageView& x, StorageView& y) const override {
        TYPE_DISPATCH(x.dtype(), compute<T>(x, y));
      }

    private:
      template <typename T>
      void compute(StorageView& x) const {
        compute::relu(x.data<T>(), x.size());
      }

      template <typename T>
      void compute(const StorageView& x, StorageView& y) const {
        y.resize_as(x);
        compute::relu(x.data<T>(), y.data<T>(), x.size());
      }
    };

    class Tanh : public UnaryOp {
    public:
      void operator()(const StorageView& x, StorageView& y) const override {
        compute<float>(x, y);
      }

    private:
      template <typename T>
      void compute(const StorageView& x, StorageView& y) const {
        y.resize_as(x);
        compute::tanh(x.data<T>(), y.data<T>(), x.size());
      }
    };

    class Sigmoid : public UnaryOp {
    public:
      void operator()(const StorageView& x, StorageView& y) const override {
        compute<float>(x, y);
      }

    private:
      template <typename T>
      void compute(const StorageView& x, StorageView& y) const {
        y = x;
        compute::mul(static_cast<T>(-1), y.data<T>(), y.size());
        compute::exp(y.data<T>(), y.data<T>(), y.size());
        compute::add(static_cast<T>(1), y.data<T>(), y.size());
        compute::inv(y.data<T>(), y.data<T>(), y.size());
      }
    };

    class SoftMax : public UnaryOp {
    public:
      void operator()(const StorageView& x, StorageView& y) const override {
        compute<float>(x, y);
      }

    private:
      template <typename T>
      void compute(const StorageView& input, StorageView& output) const {
        size_t depth = input.dim(-1);
        size_t batch_size = input.size() / depth;
        output.resize_as(input);
        for (size_t i = 0; i < batch_size; ++i) {
          const auto* x = input.data<T>() + (i * depth);
          auto* y = output.data<T>() + (i * depth);
          auto max = compute::max(x, depth);
          compute::copy(x, y, depth);
          compute::sub(max, y, depth);
          compute::exp(y, y, depth);
          auto sum = compute::sum(y, depth);
          compute::mul(1.f / (sum + EPSILON), y, depth);
        }
      }
    };

    class Cos : public UnaryOp {
    public:
      void operator()(const StorageView& x, StorageView& y) const {
        compute<float>(x, y);
      }

    private:
      template <typename T>
      void compute(const StorageView& x, StorageView& y) const {
        y.resize_as(x);
        compute::cos(x.data<T>(), y.data<T>(), x.size());
      }
    };

    class Sin : public UnaryOp {
    public:
      void operator()(const StorageView& x, StorageView& y) const {
        compute<float>(x, y);
      }

    private:
      template <typename T>
      void compute(const StorageView& x, StorageView& y) const {
        y.resize_as(x);
        compute::sin(x.data<T>(), y.data<T>(), x.size());
      }
    };

    class Add : public BinaryOp {
    public:
      void operator()(const StorageView& a, const StorageView& b, StorageView& c) const override {
        TYPE_DISPATCH(a.dtype(), compute<T>(a, b, c));
      }

    private:
      template <typename T>
      void compute(const StorageView& a, const StorageView& b, StorageView& c) const {
        if (b.size() == 1) {
          c = a;
          compute::add(b.data<T>()[0], c.data<T>(), c.size());
        } else {
          c.resize_as(a);
          compute::add(a.data<T>(), b.data<T>(), c.data<T>(), c.size());
        }
      }
    };

    class Mul : public BinaryOp {
    public:
      void operator()(const StorageView& a, const StorageView& b, StorageView& c) const override {
        TYPE_DISPATCH(a.dtype(), compute<T>(a, b, c));
      }

    private:
      template <typename T>
      void compute(const StorageView& a, const StorageView& b, StorageView& c) const {
        if (b.size() == 1) {
          c = a;
          compute::mul(b.data<T>()[0], c.data<T>(), c.size());
        } else {
          c.resize_as(a);
          compute::mul(a.data<T>(), b.data<T>(), c.data<T>(), c.size());
        }
      }
    };

    class Gather : public BinaryOp {
    public:
      Gather(int axis = 0)
        : _axis(axis) {
        if (axis != 0)
          throw std::invalid_argument("unsupported gather axis " + std::to_string(axis));
      }

      void operator()(StorageView& data, const StorageView& input) const {
        StorageView clone(data);
        operator()(clone, input, data);
      }

      void operator()(const StorageView& data,
                      const StorageView& input,
                      StorageView& output) const override {
        TYPE_DISPATCH(data.dtype(), compute<T>(data, input, output));
      }

    private:
      int _axis;

      template <typename DataType, typename IndexType = int32_t>
      void compute(const StorageView& data, const StorageView& input, StorageView& output) const {
        Shape output_shape(input.shape());
        for (size_t i = 1; i < data.rank(); ++i)
          output_shape.push_back(data.dim(i));
        output.resize(output_shape);
        size_t copy_dim = data.stride(0);
        for (size_t i = 0; i < input.size(); ++i) {
          size_t index = input.data<IndexType>()[i];
          const auto* src = data.index<DataType>({index});
          auto* dst = output.data<DataType>() + i * copy_dim;
          compute::copy(src, dst, copy_dim);
        }
      }

    };

    class Quantize : public UnaryOp {
    public:
      Quantize(float scale = 1, float shift = 0)
        : _scale(scale)
        , _shift(shift) {
      }

      void operator()(const StorageView& x, StorageView& y) const override {
        compute<float, int16_t>(x, y);
      }

    private:
      float _scale;
      float _shift;

      template <typename In, typename Out>
      void compute(const StorageView& x, StorageView& y) const {
        y.resize_as(x);
        compute::quantize(x.data<In>(), y.data<Out>(), x.size(), _scale, _shift);
      }

    };

    class Unquantize : public UnaryOp {
    public:
      Unquantize(float scale = 1, float shift = 0)
        : _scale(scale)
        , _shift(shift) {
      }

      void operator()(const StorageView& x, StorageView& y) const override {
        compute<int16_t, float>(x, y);
      }

    private:
      float _scale;
      float _shift;

      template <typename In, typename Out>
      void compute(const StorageView& x, StorageView& y) const {
        y.resize_as(x);
        compute::unquantize(x.data<In>(), y.data<Out>(), x.size(), _scale, _shift);
      }

    };

    class TopK : public Op {
    public:
      TopK(size_t k, int axis = -1)
        : _k(k)
        , _axis(axis) {
        if (axis != -1)
          throw std::invalid_argument("unsupported topk axis " + std::to_string(axis));
      }

      void operator()(const std::vector<StorageView*>& inputs,
                      std::vector<StorageView*>& outputs) const override {
        operator()(*inputs[0], *outputs[0], *outputs[1]);
      }

      void operator()(const StorageView& x, StorageView& values, StorageView& indices) const {
        compute<float, int32_t>(x, values, indices);
      }

    private:
      size_t _k;
      int _axis;

      template <typename DataType, typename IndexType>
      void compute(const StorageView& x,
                   StorageView& values,
                   StorageView& indices) const {
        size_t depth = x.dim(-1);
        size_t batch_size = x.size() / depth;
        StorageView tmp({depth}, indices.dtype());
        values.resize({batch_size, _k});
        indices.resize({batch_size, _k});
        for (size_t i = 0; i < batch_size; ++i) {
          const auto* input = x.data<DataType>() + (i * depth);
          compute::topk(input, tmp.data<IndexType>(), _k, depth);
          auto* val = values.data<DataType>() + (i * _k);
          auto* ind = indices.data<IndexType>() + (i * _k);
          compute::copy(tmp.data<IndexType>(), ind, _k);
          for (size_t j = 0; j < _k; ++j)
            val[j] = input[ind[j]];
        }
      }

    };

  }
}
