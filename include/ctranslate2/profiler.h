#pragma once

#include <chrono>
#include <ostream>
#include <string>

namespace ctranslate2 {

#ifdef ENABLE_PROFILING
#  define PROFILE(NAME) ctranslate2::Profiler profiler(NAME)
#else
#  define PROFILE(NAME) do {} while(0)
#endif

#define PROFILE_FUN PROFILE(std::string(__FILE__) + ":" + std::string(__func__))

  void init_profiling(size_t num_threads = 1);  // Not thread-safe.
  void dump_profiling(std::ostream& os);  // Not thread-safe.

  // Times of profilers created in different threads with the same name are accumulated.
  class Profiler {
  public:
    Profiler(const std::string& name);
    ~Profiler();

    const std::string& name() const;

  private:
    Profiler* _parent = nullptr;
    std::string _name;
    std::chrono::high_resolution_clock::time_point _start;
  };

}
