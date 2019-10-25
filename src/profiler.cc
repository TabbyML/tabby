#include "ctranslate2/profiler.h"

#include <algorithm>
#include <iomanip>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ctranslate2 {

  static std::chrono::high_resolution_clock::time_point global_start;
  static std::unordered_map<std::string, std::chrono::microseconds> cumulated;
  static size_t threads;
  static std::mutex mutex;

  static void assert_can_profile() {
#ifndef ENABLE_PROFILING
    throw std::runtime_error("CTranslate2 was not compiled with profiling support");
#endif
  }

  void init_profiling(size_t num_threads) {
    assert_can_profile();
    threads = num_threads;
    global_start = std::chrono::high_resolution_clock::now();
  }

  void dump_profiling(std::ostream& os) {
    if (cumulated.empty())
      return;

    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - global_start);
    total_time *= threads;

    std::vector<std::pair<std::string, std::chrono::microseconds> > sorted_cumulated;
    auto measured_time = std::chrono::microseconds::zero();
    for (const auto& pair : cumulated) {
      sorted_cumulated.emplace_back(pair);
      measured_time += pair.second;
    }

    // Include total time and non measured time in the report.
    sorted_cumulated.emplace_back("<all>", total_time);
    sorted_cumulated.emplace_back("<other>", total_time - measured_time);

    // Sort from largest to smallest accumulated time.
    std::sort(sorted_cumulated.begin(), sorted_cumulated.end(),
              [] (const std::pair<std::string, std::chrono::microseconds>& a,
                  const std::pair<std::string, std::chrono::microseconds>& b) {
                return a.second > b.second;
              });

    // Get the longest profiler name to pretty print the output.
    size_t longest_name = 0;
    for (const auto& pair : sorted_cumulated)
      longest_name = std::max(longest_name, pair.first.length());

    auto total_time_us = static_cast<double>(total_time.count());
    for (const auto& pair : sorted_cumulated) {
      const auto& name = pair.first;
      const auto& time_us = static_cast<double>(pair.second.count());
      os << std::right << std::setw(6) << std::fixed << std::setprecision(2)
         << (time_us / total_time_us) * 100 << '%'
         << ' ' << std::left << std::setw(longest_name) << name
         << ' ' << (time_us / 1000) << "ms"
         << std::endl;
    }

    cumulated.clear();
  }


  // Track active profiler in the current thread.
  static thread_local const std::string* active_profiler = nullptr;

  Profiler::Profiler(const std::string& name)
    : _name(name)
    , _start(std::chrono::high_resolution_clock::now()) {
    assert_can_profile();
    if (active_profiler)
      throw std::invalid_argument("Nested profilers are unsupported: tried to start profiler for '"
                                  + name + "' but profiler '"
                                  + *active_profiler + "' is active");
    active_profiler = &_name;
  }

  Profiler::~Profiler() {
    auto diff = std::chrono::high_resolution_clock::now() - _start;
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(diff);
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto it = cumulated.find(_name);
      if (it == cumulated.end())
        cumulated.emplace(_name, elapsed);
      else
        it->second += elapsed;
    }
    active_profiler = nullptr;
  }

}
