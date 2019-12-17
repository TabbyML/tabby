#include "ctranslate2/profiler.h"

#include <algorithm>
#include <iomanip>
#include <memory>
#include <vector>

#ifdef WITH_CUDA
#  include <cuda_runtime.h>
#endif

namespace ctranslate2 {

  static std::unique_ptr<Profiler> profiler;


  void init_profiling(Device device, size_t num_threads) {
#ifdef ENABLE_PROFILING
    profiler.reset(new Profiler(device, num_threads));
#else
    throw std::runtime_error("CTranslate2 was not compiled with profiling support");
#endif
  }

  void dump_profiling(std::ostream& os) {
    if (profiler) {
      profiler->dump(os);
      profiler.reset();
    }
  }


  Profiler::Profiler(Device device, size_t num_threads)
    : _device(device)
    , _num_threads(num_threads)
    , _global_start(std::chrono::high_resolution_clock::now()) {
  }

  Device Profiler::device() const {
    return _device;
  }

  ScopeProfile& Profiler::get_scope_profile(const std::string& name) {
    auto it = _cumulated.find(name);
    if (it == _cumulated.end())
      it = _cumulated.emplace(name, ScopeProfile()).first;
    return it->second;
  }

  void Profiler::add_scope_time(const std::string& name,
                                const std::chrono::microseconds& elapsed,
                                const std::string* parent_name) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& scope_profile = get_scope_profile(name);
    scope_profile.time_in_scope += elapsed;
    scope_profile.time_in_scope_and_callees += elapsed;
    if (parent_name) {
      auto& parent_scope_profile = profiler->get_scope_profile(*parent_name);
      parent_scope_profile.time_in_scope -= elapsed;
    }
  }

  static void print_as_percentage(std::ostream& os, double ratio) {
    os << std::right << std::setw(6) << std::fixed << std::setprecision(2)
       << ratio * 100 << '%';
  }

  void Profiler::dump(std::ostream& os) const {
    if (_cumulated.empty())
      return;

    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - _global_start);
    total_time *= _num_threads;

    // Sort from largest to smallest accumulated time.
    std::vector<std::pair<std::string, ScopeProfile>> sorted_cumulated(_cumulated.begin(),
                                                                       _cumulated.end());
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
  }


  // Track active scope in the current thread.
  static thread_local ScopeProfiler* current_scope = nullptr;

  ScopeProfiler::ScopeProfiler(const std::string& name) {
    if (!profiler)
      return;
    _parent = current_scope;
    _name = name;
    _start = std::chrono::high_resolution_clock::now();
    current_scope = this;
  }

  ScopeProfiler::~ScopeProfiler() {
    if (!profiler)
      return;
#ifdef WITH_CUDA
    if (profiler->device() == Device::CUDA)
      cudaDeviceSynchronize();
#endif
    auto diff = std::chrono::high_resolution_clock::now() - _start;
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(diff);
    profiler->add_scope_time(_name, elapsed, _parent ? &_parent->_name : nullptr);
    current_scope = _parent;
  }

}
