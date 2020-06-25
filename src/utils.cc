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

#include "cpu/backend.h"
#include "cpu/cpu_info.h"
#include "cpu/cpu_isa.h"

namespace ctranslate2 {

  bool string_to_bool(const std::string& str) {
    return str == "1" || str == "true" || str == "TRUE";
  }

  std::string read_string_from_env(const char* var, const std::string& default_value) {
    const char* value = std::getenv(var);
    if (!value)
      return default_value;
    return value;
  }

  bool read_bool_from_env(const char* var, const bool default_value) {
    return string_to_bool(read_string_from_env(var, default_value ? "1" : "0"));
  }

  bool verbose_mode() {
    static const bool verbose = read_bool_from_env("CT2_VERBOSE");
    return verbose;
  }

  static void log_config() {
    LOG() << std::boolalpha
          << "CPU: " << cpu::cpu_vendor()
          << " (SSE4.1=" << cpu::cpu_supports_sse41()
          << ", AVX=" << cpu::cpu_supports_avx()
          << ", AVX2=" << cpu::cpu_supports_avx()
          << ")" << std::endl;
    LOG() << "Selected CPU ISA: " << cpu::isa_to_str(cpu::get_cpu_isa()) << std::endl;
    LOG() << "Use Intel MKL: " << cpu::mayiuse_mkl() << std::endl;
    LOG() << "SGEMM CPU backend: "
          << cpu::gemm_backend_to_str(cpu::get_gemm_backend(ComputeType::FLOAT))
          << std::endl;
    LOG() << "GEMM_S16 CPU backend: "
          << cpu::gemm_backend_to_str(cpu::get_gemm_backend(ComputeType::INT16))
          << std::endl;
    LOG() << "GEMM_S8 CPU backend: "
          << cpu::gemm_backend_to_str(cpu::get_gemm_backend(ComputeType::INT8))
          << " (u8s8 preferred: " << cpu::prefer_u8s8s32_gemm() << ")"
          << std::endl;
    LOG() << "Use packed GEMM: " << cpu::should_pack_gemm_weights() << std::endl;
  }

  // Maybe log run configuration on program start.
  static struct ConfigLogger {
    ConfigLogger() {
      if (verbose_mode()) {
        log_config();
      }
    }
  } config_logger;

  bool mayiuse_int16(Device device, int) {
    switch (device) {
    case Device::CPU:
      return cpu::has_gemm_backend(ComputeType::INT16);
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
    case Device::CPU:
      return cpu::has_gemm_backend(ComputeType::INT8);
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
