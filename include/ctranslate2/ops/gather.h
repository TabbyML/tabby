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

      void operator()(const StorageView& data,
                      const StorageView& input,
                      StorageView& output) const override {
        DEVICE_DISPATCH(data.device(),
                        TYPE_DISPATCH(data.dtype(), (compute<D, T>(data, input, output))));
      }

    private:
      int _axis;

      template <Device D, typename DataType, typename IndexType = int32_t>
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
          primitives<D>::copy(src, dst, copy_dim);
        }
      }

    };

  }
}
