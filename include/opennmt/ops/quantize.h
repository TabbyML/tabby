#pragma once

#include "op.h"

namespace opennmt {
  namespace ops {

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
        primitives::quantize(x.data<In>(), y.data<Out>(), x.size(), _scale, _shift);
      }

    };

  }
}
