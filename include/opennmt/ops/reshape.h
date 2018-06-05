#pragma once

#include "op.h"

namespace opennmt {
  namespace ops {

    class Reshape {
    public:
      // TODO: support -1 dimension.
      void operator()(StorageView& data, const StorageView& shape) const {
        data.reshape(std::vector<size_t>(shape.data<int32_t>(),
                                         shape.data<int32_t>() + shape.size()));
      }

      void operator()(const StorageView& data,
                      const StorageView& shape,
                      StorageView& reshaped) const {
        reshaped = data;
        reshaped.reshape(std::vector<size_t>(shape.data<int32_t>(),
                                             shape.data<int32_t>() + shape.size()));
      }
    };

  }
}
