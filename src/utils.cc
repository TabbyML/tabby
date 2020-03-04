#include "ctranslate2/utils.h"

#include <sys/stat.h>
#include <chrono>
#include <cstdlib>

#ifdef _WIN32
#  include <malloc.h>
#endif

#ifdef WITH_MKL
#  include <mkl.h>
#endif

#ifdef _OPENMP
#  include <omp.h>
#endif

#ifdef WITH_CUDA
#  include "./cuda/utils.h"
#endif

namespace ctranslate2 {

#ifdef WITH_MKL
  static bool mkl_has_fast_int_gemm() {
#  if __INTEL_MKL__ > 2019 || (__INTEL_MKL__ == 2019 && __INTEL_MKL_UPDATE__ >= 5)
    // Intel MKL 2019.5 added optimized integers GEMM for SSE4.2 and AVX (in addition to
    // the existing AVX2 and AVX512), so it is virtually optimized for all target platforms.
    return true;
#  else
    return mkl_cbwr_get_auto_branch() >= MKL_CBWR_AVX2;
#  endif
  }
#endif

  bool mayiuse_int16(Device device, int) {
    switch (device) {
#ifdef WITH_MKL
    case Device::CPU:
      return mkl_has_fast_int_gemm();
#endif
    default:
      return false;
    }
  }

  bool mayiuse_int8(Device device, int device_index) {
    switch (device) {
#ifdef WITH_CUDA
    case Device::CUDA:
      return cuda::has_fast_int8(device_index);
#endif
#ifdef WITH_MKL
    case Device::CPU:
      return mkl_has_fast_int_gemm();
#endif
    default:
      return false;
    }
  }

  void set_num_threads(size_t num_threads) {
#ifdef _OPENMP
    if (num_threads != 0)
      omp_set_num_threads(num_threads);
#endif
  }

  bool ends_with(const std::string& str, const std::string& suffix) {
    return (str.size() >= suffix.size() &&
            str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0);
  }

  bool starts_with(const std::string& str, const std::string& prefix) {
    return (str.size() >= prefix.size() &&
            str.compare(0, prefix.size(), prefix) == 0);
  }

  std::vector<std::string> split_string(const std::string& str, char delimiter) {
    std::vector<std::string> parts;
    std::string part;
    for (const char c : str) {
      if (c == delimiter) {
        if (!part.empty()) {
          parts.emplace_back(std::move(part));
          part.clear();
        }
      } else {
        part += c;
      }
    }
    if (!part.empty())
      parts.emplace_back(std::move(part));
    return parts;
  }

  std::mt19937& get_random_generator() {
    static thread_local std::mt19937 generator(
      std::chrono::system_clock::now().time_since_epoch().count());
    return generator;
  }

  bool file_exists(const std::string& path) {
    struct stat buffer;
    return stat(path.c_str(), &buffer) == 0;
  }

  void* aligned_alloc(size_t size, size_t alignment) {
    void* ptr = nullptr;
#if defined(WITH_MKL)
    ptr = mkl_malloc(size, alignment);
#elif defined(_WIN32)
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0)
      ptr = nullptr;
#endif
    if (ptr == nullptr)
      throw std::runtime_error("Failed to allocate memory");
    return ptr;
  }

  void aligned_free(void* ptr) {
#if defined(WITH_MKL)
    mkl_free(ptr);
#elif defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

}
