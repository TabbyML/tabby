#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace ctranslate2 {

  bool string_to_bool(const std::string& str);

  void log_system_config();
  int get_gpu_count();
  void set_num_threads(size_t num_threads);

  bool ends_with(const std::string& str, const std::string& suffix);
  bool starts_with(const std::string& str, const std::string& prefix);

  std::vector<std::string> split_string(const std::string& str, char delimiter);
  std::vector<std::string> split_string(const std::string& str, const std::string& delimiter);
  std::string join_string(const std::vector<std::string>& tokens, const std::string& separator = "");

  std::vector<std::string> split_tokens(const std::string& text);
  std::string join_tokens(const std::vector<std::string>& tokens);

  std::vector<std::vector<std::vector<std::string>>>
  extract_features(std::vector<std::vector<std::string>> batch,
                   size_t num_features,
                   const std::string& features_separator = "￨");

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

  // Helper function to only run the model on inputs without an immediate result.
  template <typename Result, typename SkipRun, typename GetBatchResults>
  std::vector<Result>
  get_batch_results_helper(size_t batch_size,
                           const SkipRun& skip_run,
                           const GetBatchResults& get_batch_results) {
    if (batch_size == 0)
      return {};

    std::vector<Result> final_results(batch_size);
    std::vector<size_t> index_to_run;
    index_to_run.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
      if (!skip_run(i, final_results[i]))
        index_to_run.emplace_back(i);
    }

    if (!index_to_run.empty()) {
      std::vector<Result> results = get_batch_results(index_to_run);
      for (size_t i = 0; i < results.size(); ++i)
        final_results[index_to_run[i]] = std::move(results[i]);
    }

    return final_results;
  }

  template <typename T>
  inline T ceil_divide(const T& x, const T& y) {
    return (x + y - 1) / y;
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
