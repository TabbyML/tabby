#include "ctranslate2/ops/nccl_ops.h"
#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void ReduceAll::compute(const StorageView& /*input*/, StorageView& /*output*/) const {
      throw std::runtime_error("reduce all is not applied for the cpu");
    }

    template <Device D, typename T>
    void GatherAll::compute(const StorageView& /*input*/, StorageView& /*output*/) const {
      throw std::runtime_error("gather all is not applied for the cpu");
    }
    #define DECLARE_IMPL(T)                                                 \
        template void ReduceAll::compute<Device::CPU, T>(const StorageView&, \
                                                          StorageView&) const; \
        template void GatherAll::compute<Device::CPU, T>(const StorageView&, \
                                                          StorageView&) const;
    DECLARE_ALL_TYPES(DECLARE_IMPL)
  }
}
