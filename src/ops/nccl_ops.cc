#include "ctranslate2/ops/nccl_ops.h"
#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    ReduceAll::ReduceAll(ReduceAll::RED_OP op)
      : _reduce_op(op) {
    }

    void ReduceAll::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("ReduceAll");
      DEVICE_AND_TYPE_DISPATCH(input.device(), input.dtype(), (compute<D, T>(input, output)));
    }

    GatherAll::GatherAll() = default;

    void GatherAll::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("ReduceAll");
      DEVICE_AND_TYPE_DISPATCH(input.device(), input.dtype(), (compute<D, T>(input, output)));
    }
  }
}
