#include "ctranslate2/ops/op.h"

namespace ctranslate2 {
  namespace ops {

    bool are_broadcastable(const StorageView& x, const StorageView& y, Shape* output_shape) {
      if (x.rank() != y.rank())
        return false;
      if (output_shape)
        output_shape->clear();
      const Shape& x_shape = x.shape();
      const Shape& y_shape = y.shape();
      for (size_t i = 0; i < x_shape.size(); ++i) {
        size_t x_dim = x_shape[i];
        size_t y_dim = y_shape[i];
        if (x_dim != y_dim && x_dim != 1 && y_dim != 1)
          return false;
        if (output_shape)
          output_shape->push_back(std::max(x_dim, y_dim));
      }
      return true;
    }

    Shape get_dim_repeats(const Shape& shape, const Shape& full_shape) {
      Shape repeats;
      repeats.reserve(shape.size());
      for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 1 && shape[i] != full_shape[i])
          repeats.push_back(full_shape[i]);
        repeats.push_back(1);
      }
      return repeats;
    }

  }
}
