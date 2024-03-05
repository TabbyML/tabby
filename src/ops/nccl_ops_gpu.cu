#include "ctranslate2/ops/nccl_ops.h"
#ifdef CT2_WITH_TENSOR_PARALLEL
  #include <nccl.h>
  #include "cuda/utils.h"
#endif
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

#ifdef CT2_WITH_TENSOR_PARALLEL
    ncclDataType_t getNcclDataTypeFromDataType(DataType type) {
      switch (type) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0)
        case DataType::BFLOAT16:
          return ncclBfloat16;
#endif
        case DataType::FLOAT16:
          return ncclFloat16;
        case DataType::FLOAT32:
          return ncclFloat32;
        case DataType::INT32:
          return ncclInt32;
        case DataType::INT8:
          return ncclInt8;
        default:
          throw std::invalid_argument("The current datatype " + std::to_string(static_cast<int>(type)) +
                                      " is not supported for the mode tensor parallel ");
      }
    }

    ncclRedOp_t redop_to_nccl_op(ReduceAll::RED_OP op) {
      switch (op) {
        case ReduceAll::RED_OP::SUM:
          return ncclSum;
        case ReduceAll::RED_OP::PROD:
          return ncclProd;
        case ReduceAll::RED_OP::MAX:
          return ncclMax;
        case ReduceAll::RED_OP::MIN:
          return ncclMin;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,10,0)
        case ReduceAll::RED_OP::AVG:
          return ncclAvg;
#endif
        default:
          throw std::runtime_error("the current reduce operation " + std::to_string(static_cast<int>(op)) + " is not supported");
      }
    }
#endif

    template <Device D, typename T>
    void ReduceAll::compute(const StorageView& input, StorageView& output) const {
#ifdef CT2_WITH_TENSOR_PARALLEL
      // initializing NCCL
      dim_t data_size = input.size();
      ncclComm_t comm = ScopedMPISetter::getNcclComm();
      ncclDataType_t ncclDataType = getNcclDataTypeFromDataType(input.dtype());
      ncclRedOp_t ncclOp = redop_to_nccl_op(_reduce_op);
      NCCL_CHECK(ncclAllReduce(input.data<T>(), output.data<T>(),
                               data_size, ncclDataType, ncclOp,
                               comm, cuda::get_cuda_stream()));

      cudaStreamSynchronize(cuda::get_cuda_stream());
#endif
      (void)input;
      (void)output;
    }

    template <Device D, typename T>
    void GatherAll::compute(const StorageView& input, StorageView& output) const {
#ifdef CT2_WITH_TENSOR_PARALLEL
      // initializing NCCL
      dim_t data_size = input.size();
      ncclComm_t comm = ScopedMPISetter::getNcclComm();
      ncclDataType_t ncclDataType = getNcclDataTypeFromDataType(input.dtype());
      NCCL_CHECK(ncclAllGather(input.data<T>(), output.data<T>(),
                               data_size, ncclDataType,
                               comm, cuda::get_cuda_stream()));

      cudaStreamSynchronize(cuda::get_cuda_stream());
#endif
      (void)input;
      (void)output;
    }
#define DECLARE_IMPL(T)                                                 \
        template void GatherAll::compute<Device::CUDA, T>(const StorageView&, \
                                                          StorageView&) const; \
        template void ReduceAll::compute<Device::CUDA, T>(const StorageView&, \
                                                          StorageView&) const;
    DECLARE_ALL_TYPES(DECLARE_IMPL)
  }
}
