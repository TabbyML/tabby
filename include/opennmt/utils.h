#pragma once

#include <cstddef>

namespace opennmt {

  // Global utility functions.

  // Intitializes the library. Sets the number of threads and tune some MKL modes.
  void init(size_t num_threads = 0);

  // Sets the global number of threads to use.
  void set_num_threads(size_t num_threads);

}
