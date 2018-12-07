#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Unquantize : public BinaryOp {
    public:
      void operator()(const StorageView& x, const StorageView& scale, StorageView& y) const override {
        TYPE_DISPATCH(x.dtype(), (compute<T, float>(x, scale, y)));
      }

    private:
      template <typename In, typename Out>
      void compute(const StorageView& x, const StorageView& scale, StorageView& y) const {
        if (x.device() == Device::CUDA)
          throw std::invalid_argument("Unquantize op is only defined on CPU for now");
        y.resize_as(x);
        if (scale.is_scalar())
          primitives<>::unquantize(x.data<In>(), y.data<Out>(), x.size(), scale.as_scalar<Out>());
        else {
          size_t depth = x.dim(-1);
          size_t batch_size = x.size() / depth;

          const auto* scale_data = scale.data<Out>();
          const auto* x_data = x.data<In>();
          auto* y_data = y.data<Out>();
          if (scale.size() == batch_size) {  // Per-batch scale.
            for (size_t b = 0; b < batch_size; ++b) {
              for (size_t i = 0; i < depth; ++i) {
                size_t index = b * depth + i;
                y_data[index] = static_cast<Out>(x_data[index]) / scale_data[b];
              }
            }
          } else {
            throw std::invalid_argument("unsupported quantization scale shape");
          }
        }
      }

    };

  }
}
