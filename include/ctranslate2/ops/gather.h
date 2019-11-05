#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Gather : public BinaryOp {
    public:
      Gather(int axis = 0);
      using BinaryOp::operator();

      void operator()(StorageView& data, const StorageView& input) const;
      void operator()(const StorageView& data,
                      const StorageView& input,
                      StorageView& output) const override;

    private:
      template <Device D, typename T>
      void compute(const StorageView& data, const StorageView& input, StorageView& output) const;
    };

  }
}
