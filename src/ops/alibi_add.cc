#include "ctranslate2/ops/alibi_add.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    AlibiAdd::AlibiAdd(const bool use_positive_positions)
      : _use_positive_positions(use_positive_positions)
    {
    }

    void AlibiAdd::operator()(const StorageView& input,
                              const StorageView& alibi,
                              StorageView& output) const {
      PROFILE("AlibiAdd");

      output.resize_as(input);

      const Device device = input.device();
      const DataType dtype = input.dtype();
      const dim_t alibi_offset = _use_positive_positions ? 0 : alibi.dim(-1) - input.dim(-1);

      switch (dtype) {
      case DataType::FLOAT32: {
        DEVICE_DISPATCH(device, (compute<D, float>(input, alibi, alibi_offset, output)));
        break;
      }
#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16: {
        if (device != Device::CUDA)
          throw std::invalid_argument("FP16 AlibiAdd is only supported on GPU");
        compute<Device::CUDA, float16_t>(input, alibi, alibi_offset, output);
        break;
      }
#endif
      default:
        throw std::invalid_argument("AlibiAdd only supports float types");
      }
    }

  }
}
