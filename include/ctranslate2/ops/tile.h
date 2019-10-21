#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Tile : public BinaryOp {
    public:
      void operator()(const StorageView& input,
                      const StorageView& repeats,
                      StorageView& output) const override;

    private:
      template <Device D, typename T>
      void compute(const StorageView& input,
                   const StorageView& repeats,
                   StorageView& output) const {
        assert(repeats.size() == input.rank());

        Shape output_shape(input.shape());
        size_t last_repeated_dim = 0;
        for (size_t i = 0; i < output_shape.size(); ++i) {
          const size_t repeat_dim = repeats.at<int32_t>(i);
          if (repeat_dim != 1) {
            last_repeated_dim = i;
            output_shape[i] *= repeat_dim;
          }
        }

        output.resize(output_shape);

        size_t copied = 0;
        size_t i = last_repeated_dim;
        do {
          const size_t axis = i;
          const size_t repeat_dim = repeats.at<int32_t>(axis);
          size_t iter_dim = 1;
          size_t copy_dim = 1;
          for (size_t k = 0; k < axis; ++k)
            iter_dim *= input.dim(k);
          for (size_t k = axis; k < input.rank(); ++k)
            copy_dim *= input.dim(k);
          if (axis == last_repeated_dim) {
            for (size_t j = 0; j < iter_dim; ++j) {
              for (size_t r = 0; r < repeat_dim; ++r) {
                primitives<D>::copy(input.data<T>() + j * copy_dim,
                                    output.data<T>() + copied, copy_dim);
                copied += copy_dim;
              }
            }
          } else {
            for (size_t r = 1; r < repeat_dim; ++r) {
              primitives<D>::copy(output.data<T>(),
                                  output.data<T>() + r * copied, copied);
            }
            copied *= repeat_dim;
          }
        } while (i-- > 0);
      }
    };

  }
}
