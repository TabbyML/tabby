#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class GELU : public UnaryOp {
    public:
      enum class Approximation {
        None,
        Tanh,
        Sigmoid,
      };

      GELU(const Approximation approximation = Approximation::None);

      void operator()(const StorageView& x, StorageView& y) const override;

    private:
      template <Device D, typename T>
      void compute(const StorageView& x, StorageView& y) const {
        switch (_approximation) {
        case Approximation::None:
          primitives<D>::gelu(x.data<T>(), y.data<T>(), x.size());
          break;
        case Approximation::Tanh:
          primitives<D>::gelu_tanh(x.data<T>(), y.data<T>(), x.size());
          break;
        case Approximation::Sigmoid:
          primitives<D>::gelu_sigmoid(x.data<T>(), y.data<T>(), x.size());
          break;
        }
      }

      const Approximation _approximation;
    };

  }
}
