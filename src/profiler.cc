#include "ctranslate2/profiler.h"

#include <algorithm>
#include <iomanip>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef WITH_CUDA
#  include <cuda_runtime.h>
#endif

namespace ctranslate2 {

  struct ScopeProfile {
    std::chrono::microseconds time_in_scope;
    std::chrono::microseconds time_in_scope_and_callees;
  };

  static std::chrono::high_resolution_clock::time_point global_start;
  static std::unordered_map<std::string, ScopeProfile> cumulated;
  static size_t threads;
  static std::mutex mutex;
  static bool do_profile = false;

  static void assert_can_profile() {
#ifndef ENABLE_PROFILING
    throw std::runtime_error("CTranslate2 was not compiled with profiling support");
#endif
  }

  static void print_as_percentage(std::ostream& os, double ratio) {
    os << std::right << std::setw(6) << std::fixed << std::setprecision(2)
       << ratio * 100 << '%';
  }

  static ScopeProfile& get_scope_profile(const std::string& name) {
    auto it = cumulated.find(name);
    if (it == cumulated.end())
      it = cumulated.emplace(name, ScopeProfile()).first;
    return it->second;
  }

  void init_profiling(size_t num_threads) {
    assert_can_profile();
    threads = num_threads;
    global_start = std::chrono::high_resolution_clock::now();
    do_profile = true;
  }

  void dump_profiling(std::ostream& os) {
    if (cumulated.empty())
      return;

    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - global_start);
    total_time *= threads;

    // Sort from largest to smallest accumulated time.
    std::vector<std::pair<std::string, ScopeProfile>> sorted_cumulated(cumulated.begin(),
                                                                       cumulated.end());
    std::sort(sorted_cumulated.begin(), sorted_cumulated.end(),
              [] (const std::pair<std::string, ScopeProfile>& a,
                  const std::pair<std::string, ScopeProfile>& b) {
                return a.second.time_in_scope > b.second.time_in_scope;
              });

    // Get the longest profiler name to pretty print the output.
    size_t longest_name = 0;
    for (const auto& pair : sorted_cumulated)
      longest_name = std::max(longest_name, pair.first.length());

    double total_time_us = total_time.count();
    double ratio_printed_so_far = 0;
    for (const auto& pair : sorted_cumulated) {
      const auto& name = pair.first;
      const auto& result = pair.second;

      double time_in_scope_us = result.time_in_scope.count();
      double time_in_scope_and_callees_us = result.time_in_scope_and_callees.count();
      double time_in_scope_ratio = time_in_scope_us / total_time_us;
      double time_in_scope_and_callees_ratio = time_in_scope_and_callees_us / total_time_us;
      ratio_printed_so_far += time_in_scope_ratio;

      print_as_percentage(os, time_in_scope_ratio);
      os << ' ';
      print_as_percentage(os, time_in_scope_and_callees_ratio);
      os << ' ';
      print_as_percentage(os, ratio_printed_so_far);
      os << ' ' << std::left << std::setw(longest_name) << name
         << ' ' << (time_in_scope_us / 1000) << "ms"
         << std::endl;
    }

    cumulated.clear();
    do_profile = false;
  }


  // Track active profiler in the current thread.
  static thread_local Profiler* current_profiler = nullptr;

  Profiler::Profiler(const std::string& name) {
    if (!do_profile)
      return;
    _parent = current_profiler;
    _name = name;
    _start = std::chrono::high_resolution_clock::now();
    current_profiler = this;
  }

  Profiler::~Profiler() {
    if (!do_profile)
      return;
#ifdef WITH_CUDA
    cudaDeviceSynchronize();
#endif
    auto diff = std::chrono::high_resolution_clock::now() - _start;
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(diff);
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto& scope_profile = get_scope_profile(_name);
      scope_profile.time_in_scope += elapsed;
      scope_profile.time_in_scope_and_callees += elapsed;
      if (_parent) {
        auto& parent_scope_profile = get_scope_profile(_parent->_name);
        parent_scope_profile.time_in_scope -= elapsed;
      }
    }
    current_profiler = _parent;
  }

  const std::string& Profiler::name() const {
    return _name;
  }

}
