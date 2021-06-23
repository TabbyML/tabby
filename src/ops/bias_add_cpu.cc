#include "ctranslate2/ops/bias_add.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void BiasAdd::compute(const StorageView& value,
                          const StorageView& bias,
                          StorageView& output) const {
      primitives<D>::add_batch_broadcast(bias.data<T>(),
                                         value.data<T>(),
                                         output.data<T>(),
                                         bias.size(),
                                         value.size());
      if (_activation_type)
        get_activation_op(*_activation_type)(output, output);
    }

#define DECLARE_IMPL(T)                                         \
    template void                                               \
    BiasAdd::compute<Device::CPU, T>(const StorageView& value,  \
                                     const StorageView& bias,   \
                                     StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
