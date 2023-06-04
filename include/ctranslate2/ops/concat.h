#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Concat : public Op {
    public:
      Concat(int axis);
      void operator()(const std::vector<const StorageView*>& inputs,
                      StorageView& output) const;

    private:
      int _axis;

      template <Device D, typename T>
      void compute(const std::vector<const StorageView*>& inputs, StorageView& output) const;
    };

  }
}
