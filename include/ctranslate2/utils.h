#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace ctranslate2 {

  bool string_to_bool(const std::string& str);

  void set_num_threads(size_t num_threads);

  bool ends_with(const std::string& str, const std::string& suffix);
  bool starts_with(const std::string& str, const std::string& prefix);

  std::vector<std::string> split_string(const std::string& str, char delimiter);
  std::vector<std::string> split_string(const std::string& str, const std::string& delimiter);

  std::vector<std::string> split_tokens(const std::string& text);
  std::string join_tokens(const std::vector<std::string>& tokens);

  template <typename Stream>
  Stream open_file(const std::string& path) {
    Stream stream(path);
    if (!stream)
      throw std::runtime_error("Failed to open file: " + path);
    return stream;
  }

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

#ifdef NDEBUG
#  define THROW_EXCEPTION(EXCEPTION, MESSAGE) throw EXCEPTION(MESSAGE)
#else
#  define THROW_EXCEPTION(EXCEPTION, MESSAGE)                           \
  throw EXCEPTION(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + MESSAGE)
#endif
#define THROW_RUNTIME_ERROR(MESSAGE) THROW_EXCEPTION(std::runtime_error, MESSAGE)
#define THROW_INVALID_ARGUMENT(MESSAGE) THROW_EXCEPTION(std::invalid_argument, MESSAGE)

}
