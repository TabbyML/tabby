#pragma once

#include "devices.h"

namespace ctranslate2 {

  // Check feature support.
  bool mayiuse_int16(Device device);
  bool mayiuse_int8(Device device);

  void set_num_threads(size_t num_threads);

}
