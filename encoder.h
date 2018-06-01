#pragma once

#include "storage_view.h"

namespace onmt {

  class Encoder {
  public:
    virtual ~Encoder() = default;

    virtual onmt::StorageView& encode(const onmt::StorageView& ids,
                                      const onmt::StorageView& lengths) = 0;
  };

}
