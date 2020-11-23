#pragma once

#include "ops/gather.h"
#include "storage_view.h"

namespace ctranslate2 {

  // This class can be used to dynamically remove or add padding.
  // This is useful to save on computation when lengths are very different.
  class Padder {
  public:
    static inline bool allow_padding_removal(const Device device,
                                             const ComputeType compute_type) {
      return device == Device::CPU || compute_type != ComputeType::FLOAT16;
    }

    // If max_time is negative, it is set to the maximum length.
    Padder(const StorageView& lengths,
           const dim_t max_time = -1,
           const dim_t pad_batch_to_multiple = 1);

    // Merge batch and time dimensions and remove padding.
    void remove_padding(StorageView& x) const;

    // Split first dimension into batch and time dimensions and add padding.
    void add_padding(StorageView& x) const;

  private:
    dim_t _batch_size;
    dim_t _max_time;
    StorageView _padded_to_flat;
    StorageView _flat_to_padded;
    const ops::Gather _gather_op;
  };

}
