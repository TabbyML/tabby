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
        y.resize_as(x);
        if (scale.is_scalar())
          primitives<>::unquantize(x.data<In>(), y.data<Out>(), x.size(), scale.as_scalar<Out>());
        else
          throw std::invalid_argument("unsupported non scalar quantization scale");
      }

    };

  }
}
