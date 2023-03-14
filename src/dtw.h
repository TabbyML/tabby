#pragma once

#include "ctranslate2/storage_view.h"

namespace ctranslate2 {

  // Dynamic time wrapping function, but x values are negated.
  std::vector<std::pair<size_t, size_t>> negative_dtw(const StorageView& x);

}
