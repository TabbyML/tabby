#include "ctranslate2/ops/softmax.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void SoftMax::compute(const StorageView& input, StorageView& output) const {
      // TODO: GPU impl.

      static thread_local StorageView input_host;
      static thread_local StorageView output_host;
      input_host = input;
      compute<Device::CPU, float>(input_host, output_host);
      output = output_host;
    }

#define DECLARE_IMPL(T)                                         \
    template void                                               \
    SoftMax::compute<Device::CUDA, T>(const StorageView& input, \
                                      StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
