#pragma once

#include <chrono>
#include <mutex>
#include <ostream>
#include <string>
#include <unordered_map>

#include "ctranslate2/devices.h"

namespace ctranslate2 {

#ifdef ENABLE_PROFILING
#  define PROFILE(NAME) ctranslate2::ScopeProfiler scope_profiler(NAME)
#else
#  define PROFILE(NAME) do {} while(0)
#endif

#define PROFILE_FUN PROFILE(std::string(__FILE__) + ":" + std::string(__func__))

  void init_profiling(Device device, size_t num_threads = 1);  // Not thread-safe.
  void dump_profiling(std::ostream& os);  // Not thread-safe.


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

  class ScopeProfile {
  public:
    std::chrono::microseconds time_in_scope;
    std::chrono::microseconds time_in_scope_and_callees;
  };

  class Profiler {
  public:
    Profiler(Device device, size_t num_threads);
    Device device() const;
    void dump(std::ostream& os) const;
    void add_scope_time(const std::string& name,
                        const std::chrono::microseconds& elapsed,
                        const std::string* parent_name);

  private:
    ScopeProfile& get_scope_profile(const std::string& name);
    Device _device;
    size_t _num_threads;
    std::chrono::high_resolution_clock::time_point _global_start;
    std::unordered_map<std::string, ScopeProfile> _cumulated;
    std::mutex _mutex;
  };

}
