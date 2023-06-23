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

      const dim_t alibi_offset = _use_positive_positions ? 0 : alibi.dim(-1) - input.dim(-1);

      DEVICE_AND_FLOAT_DISPATCH("AlibiAdd", input.device(), input.dtype(),
                                (compute<D, T>(input, alibi, alibi_offset, output)));
    }

  }
}
