#pragma once

#include "storage_view.h"

namespace opennmt {

  class Encoder {
  public:
    virtual ~Encoder() = default;

    virtual StorageView& encode(const StorageView& ids,
                                const StorageView& lengths) = 0;
  };

}
