#pragma once

#include <random>
#include <string>
#include <thread>
#include <vector>

#include "devices.h"
#include "types.h"

namespace ctranslate2 {

  bool string_to_bool(const std::string& str);
  std::string read_string_from_env(const char* var, const std::string& default_value = "");
  bool read_bool_from_env(const char* var, const bool default_value = false);
  int read_int_from_env(const char* var, const int default_value = 0);

  // Check feature support.
  bool mayiuse_float16(Device device, int device_index = 0);
  bool mayiuse_int16(Device device, int device_index = 0);
  bool mayiuse_int8(Device device, int device_index = 0);
  dim_t get_preferred_size_multiple(ComputeType compute_type,
                                    Device device,
                                    int device_index = 0);

  int get_gpu_count();

  void set_num_threads(size_t num_threads);
  void set_thread_affinity(std::thread& thread, int index);

  bool ends_with(const std::string& str, const std::string& suffix);
  bool starts_with(const std::string& str, const std::string& prefix);

  std::vector<std::string> split_string(const std::string& str, char delimiter);
  std::vector<std::string> split_string(const std::string& str, const std::string& delimiter);

  template <typename T, typename I>
  static std::vector<T> index_vector(const std::vector<T>& v,
                                     const std::vector<I>& index) {
    std::vector<T> new_v;
    new_v.resize(index.size());
    for (size_t i = 0; i < index.size(); ++i)
      new_v[i] = v[index[i]];
    return new_v;
  }

  template <typename T>
  void truncate_sequences(std::vector<std::vector<T>>& sequences, size_t max_length) {
    for (auto& sequence : sequences) {
      if (sequence.size() > max_length)
        sequence.resize(max_length);
    }
  }

  void set_random_seed(const unsigned int seed);
  unsigned int get_random_seed();
  std::mt19937& get_random_generator();

#ifdef NDEBUG
#  define THROW_EXCEPTION(EXCEPTION, MESSAGE) throw EXCEPTION(MESSAGE)
#else
#  define THROW_EXCEPTION(EXCEPTION, MESSAGE)                           \
  throw EXCEPTION(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + MESSAGE)
#endif
#define THROW_RUNTIME_ERROR(MESSAGE) THROW_EXCEPTION(std::runtime_error, MESSAGE)
#define THROW_INVALID_ARGUMENT(MESSAGE) THROW_EXCEPTION(std::invalid_argument, MESSAGE)

}
