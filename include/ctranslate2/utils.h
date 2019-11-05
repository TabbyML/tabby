#pragma once

#include "devices.h"

namespace ctranslate2 {

  // Check feature support.
  bool mayiuse_int16(Device device);
  bool mayiuse_int8(Device device);

  void set_num_threads(size_t num_threads);

  bool ends_with(const std::string& str, const std::string& suffix);
  bool starts_with(const std::string& str, const std::string& prefix);

#define THROW_EXCEPTION(EXCEPTION, MESSAGE)                             \
  throw EXCEPTION(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + MESSAGE)
#define THROW_RUNTIME_ERROR(MESSAGE) THROW_EXCEPTION(std::runtime_error, MESSAGE)
#define THROW_INVALID_ARGUMENT(MESSAGE) THROW_EXCEPTION(std::invalid_argument, MESSAGE)

}
