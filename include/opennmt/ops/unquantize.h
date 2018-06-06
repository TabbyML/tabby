#pragma once

#include "op.h"

namespace opennmt {
  namespace ops {

    class Unquantize : public UnaryOp {
    public:
      Unquantize(float scale = 1, float shift = 0)
        : _scale(scale)
        , _shift(shift) {
      }

      void operator()(const StorageView& x, StorageView& y) const override {
        TYPE_DISPATCH(x.dtype(), SINGLE_ARG(compute<T, float>(x, y)));
      }

    private:
      float _scale;
      float _shift;

      template <typename In, typename Out>
      void compute(const StorageView& x, StorageView& y) const {
        y.resize_as(x);
        primitives::unquantize(x.data<In>(), y.data<Out>(), x.size(), _scale, _shift);
      }

    };

  }
}
