#pragma once

#include <chrono>
#include <ostream>
#include <string>

#include "ctranslate2/devices.h"

namespace ctranslate2 {

#ifdef CT2_ENABLE_PROFILING
#  define PROFILE(NAME) ctranslate2::ScopeProfiler scope_profiler(NAME)

  // Times of profilers created in different threads with the same name are accumulated.
  class ScopeProfiler {
  public:
    ScopeProfiler(const std::string& name);
    ~ScopeProfiler();

  private:
    ScopeProfiler* _parent = nullptr;
    std::string _name;
    std::chrono::high_resolution_clock::time_point _start;
  };

#else
#  define PROFILE(NAME) do {} while(0)
#endif

#define PROFILE_FUN PROFILE(std::string(__FILE__) + ":" + std::string(__func__))

  void init_profiling(Device device, size_t num_threads = 1);  // Not thread-safe.
  void dump_profiling(std::ostream& os);  // Not thread-safe.

}
