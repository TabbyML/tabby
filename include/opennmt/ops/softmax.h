#pragma once

#include "op.h"

namespace opennmt {
  namespace ops {

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
        output = input;
        for (size_t i = 0; i < batch_size; ++i) {
          const auto* x = input.data<T>() + (i * depth);
          auto* y = output.data<T>() + (i * depth);
          auto max = primitives::max(x, depth);
          primitives::sub(max, y, depth);
          primitives::exp(y, y, depth);
          auto sum = primitives::sum(y, depth);
          primitives::mul(1.f / (sum + EPSILON), y, depth);
        }
      }
    };

  }
}
