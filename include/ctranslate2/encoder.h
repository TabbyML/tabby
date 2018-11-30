#pragma once

#include "storage_view.h"

namespace ctranslate2 {

  // Base class for encoders.
  class Encoder {
  public:
    virtual ~Encoder() = default;

    virtual void operator()(const StorageView& ids,
                            const StorageView& lengths,
                            StorageView& output) = 0;
  };

}
