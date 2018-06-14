#pragma once

#include <cstddef>

namespace opennmt {

  void init(size_t num_threads = 0);
  void set_num_threads(size_t num_threads);

}
