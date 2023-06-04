#pragma once

#include <algorithm>

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Squeeze : public UnaryOp {
    public:
      Squeeze(const std::vector<size_t>& axes)
        : _axes(axes) {
        std::sort(_axes.begin(), _axes.end());
      }

      void operator()(StorageView& data) const {
        PROFILE("Squeeze");
        data.reshape(transform_shape(data.shape()));
      }
      void operator()(const StorageView& data, StorageView& squeezed) const override {
        PROFILE("Squeeze");
        squeezed.shallow_copy(const_cast<StorageView&>(data));
        squeezed.reshape(transform_shape(data.shape()));
      }

    private:
      std::vector<size_t> _axes;

      Shape transform_shape(const Shape& shape) const {
        Shape new_shape;
        for (size_t i = 0, j = 0; i < shape.size(); ++i) {
          if (j < _axes.size() && i == _axes[j]) {
            if (shape[i] != 1)
              throw std::invalid_argument("can't squeeze dimension " + std::to_string(i)
                                          + " which is " + std::to_string(shape[i])
                                          + ", but should be 1");
            ++j;
          } else {
            new_shape.push_back(shape[i]);
          }
        }
        return new_shape;
      }

    };

  }
}
