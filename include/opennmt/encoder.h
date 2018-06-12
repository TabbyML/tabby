#pragma once

#include "storage_view.h"

namespace opennmt {

  class Encoder {
  public:
    virtual ~Encoder() = default;

    virtual void encode(const StorageView& ids,
                        const StorageView& lengths,
                        StorageView& output) = 0;
  };

}
