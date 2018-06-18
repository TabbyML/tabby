#pragma once

#include "storage_view.h"

namespace opennmt {

  // Base class for encoders.
  class Encoder {
  public:
    virtual ~Encoder() = default;

    virtual void encode(const StorageView& ids,
                        const StorageView& lengths,
                        StorageView& output) = 0;
  };

}
