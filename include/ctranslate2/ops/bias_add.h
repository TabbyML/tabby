#pragma once

#include "activation.h"
#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class BiasAdd : public BinaryOp {
    public:
      BiasAdd(const ActivationType* activation_type = nullptr);

      void operator()(const StorageView& value,
                      const StorageView& bias,
                      StorageView& output) const override;

    private:
      template <Device D, typename T>
      void compute(const StorageView& value,
                   const StorageView& bias,
                   StorageView& output) const;

      const ActivationType* _activation_type;
    };

  }
}
