#pragma once

#include <limits>

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class TopPMask : public Op {
    public:
      TopPMask(const float p, const float mask_value = -std::numeric_limits<float>::infinity());

      static dim_t max_num_classes(const Device device);

      void operator()(const StorageView& input, StorageView& output) const;

    private:
      const float _p;
      const float _mask_value;

      template <Device D>
      static dim_t max_num_classes();

      template <Device D, typename T>
      void compute(const StorageView& input, const StorageView& probs, StorageView& output) const;
    };

  }
}
