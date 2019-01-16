#pragma once

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace layers {

    // Base class for encoders.
    class Encoder {
    public:
      virtual ~Encoder() = default;

      virtual void operator()(const StorageView& ids,
                              const StorageView& lengths,
                              StorageView& output) = 0;
    };

  }
}
