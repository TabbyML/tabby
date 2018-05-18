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

    class Linear {
    public:
      Linear(const StorageView<float>& weight, const StorageView<float>* bias = nullptr)
        : _weight(weight)
        , _bias(bias) {
      }

      void operator()(const StorageView<float>& input, StorageView<float>& output) const {
        size_t input_depth = input.dim(-1);
        size_t m = input.size() / input_depth;
        size_t n = output_depth();
        size_t k = input_depth;
        assert(k == _weight.dim(1));

        output.resize({m, n});

        float beta = 0.0;
        if (_bias != nullptr) {
          beta = 1.0;
          for (size_t i = 0; i < m; ++i)
            array_copy(_bias->data(), output.index({i}), n);
        }

        sgemm(input.data(), _weight.data(),
              CblasNoTrans, CblasTrans,
              m, n, k,
              beta, output.data());

        // Restore collapsed dimensions.
        Shape output_shape(input.shape());
        output_shape.back() = n;
        output.reshape(output_shape);
      }

      size_t output_depth() const {
        return _weight.dim(0);
      }

    private:
      const StorageView<float>& _weight;
      const StorageView<float>* _bias;
    };

    class ReLU {
    public:
      void operator()(StorageView<float>& x) const {
        for (size_t i = 0; i < x.size(); ++i) {
          if (x[i] < 0)
            x[i] = 0;
        }
      }

      void operator()(const StorageView<float>& input, StorageView<float>& output) const {
        output = input;
        operator()(output);
      }
    };

    class SoftMax {
    public:
      void operator()(const StorageView<float>& input, StorageView<float>& output) const {
        size_t depth = input.dim(-1);
        size_t batch_size = input.size() / depth;
        for (size_t i = 0; i < batch_size; ++i) {
          const float* x = input.data() + (i * depth);
          float* y = output.data() + (i * depth);
          float max = array_max(x, depth);
          array_copy(x, y, depth);
          array_sub(max, y, depth);
          array_exp(y, y, depth);
          float sum = array_sum(y, depth);
          array_mul(1.f / (sum + EPSILON), y, depth);
        }
      }
    };

    class Gather {
    public:
      Gather(const StorageView<float>& from)
        : _from(from) {
        assert(from.rank() == 2);
      }

      void operator()(const StorageView<size_t>& input, StorageView<float>& output) const {
        size_t batch_size = input.dim(0);
        size_t depth = _from.dim(-1);
        output.resize({batch_size, depth});
        for (size_t i = 0; i < batch_size; ++i) {
          const float* src = _from.index({input[i]});
          float* dst = output.index({i});
          array_copy(src, dst, depth);
        }
      }

      size_t output_depth() const {
        return _from.dim(-1);
      }

    private:
      const StorageView<float>& _from;
    };

  }
}
