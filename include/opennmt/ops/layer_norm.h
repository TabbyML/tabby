#pragma once

#include "opennmt/ops/op.h"

#define EPSILON 0.000001f

namespace opennmt {
  namespace ops {

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
          const auto* x = input.data<T>() + i * depth;
          auto* y = output.data<T>() + i * depth;
          auto mean = primitives::mean(x, depth);
          primitives::copy(x, y, depth);
          primitives::sub(mean, y, depth);
          primitives::pow(y, tmp.data<T>(), static_cast<T>(2), depth);
          auto variance = primitives::mean(tmp.data<T>(), depth);
          primitives::mul(static_cast<T>(1.0 / sqrt(variance + EPSILON)), y, depth);
          primitives::mul(gamma.data<T>(), y, depth);
          primitives::add(beta.data<T>(), y, depth);
        }
      }
    };

  }
}
