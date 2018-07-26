#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Gather : public BinaryOp {
    public:
      Gather(int axis = 0)
        : _axis(axis) {
        if (axis != 0)
          throw std::invalid_argument("unsupported gather axis " + std::to_string(axis));
      }

      using BinaryOp::operator();
      void operator()(const StorageView& data,
                      const StorageView& input,
                      StorageView& output) const override {
        Shape output_shape(input.shape());
        for (size_t i = 1; i < data.rank(); ++i)
          output_shape.push_back(data.dim(i));
        output.resize(output_shape);
        DEVICE_DISPATCH(data.device(),
                        TYPE_DISPATCH(data.dtype(), (compute<D, T>(data, input, output))));
      }

    private:
      int _axis;

      template <Device D, typename T>
      void compute(const StorageView& data, const StorageView& input, StorageView& output) const;

    };

  }
}
