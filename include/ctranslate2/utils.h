#pragma once

#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "devices.h"

namespace ctranslate2 {

  bool string_to_bool(const std::string& str);
  std::string read_string_from_env(const char* var, const std::string& default_value = "");
  bool read_bool_from_env(const char* var, const bool default_value = false);
  int read_int_from_env(const char* var, const int default_value = 0);

  bool verbose_mode();

  // Check feature support.
  bool mayiuse_float16(Device device, int device_index = 0);
  bool mayiuse_int16(Device device, int device_index = 0);
  bool mayiuse_int8(Device device, int device_index = 0);

  void set_num_threads(size_t num_threads);

  bool ends_with(const std::string& str, const std::string& suffix);
  bool starts_with(const std::string& str, const std::string& prefix);

  std::vector<std::string> split_string(const std::string& str, char delimiter);

  std::mt19937& get_random_generator();

  void* aligned_alloc(size_t size, size_t alignment);
  void aligned_free(void* ptr);

#ifdef NDEBUG
#  define THROW_EXCEPTION(EXCEPTION, MESSAGE) throw EXCEPTION(MESSAGE)
#else
#  define THROW_EXCEPTION(EXCEPTION, MESSAGE)                           \
  throw EXCEPTION(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + MESSAGE)
#endif
#define THROW_RUNTIME_ERROR(MESSAGE) THROW_EXCEPTION(std::runtime_error, MESSAGE)
#define THROW_INVALID_ARGUMENT(MESSAGE) THROW_EXCEPTION(std::invalid_argument, MESSAGE)

#define LOG() if (verbose_mode()) std::cerr << "[ct2_verbose] "

}
