#include "ctranslate2/utils.h"

#include <cstdlib>

#ifdef _OPENMP
#  include <omp.h>
#endif

#ifdef CT2_WITH_CUDA
#  include "./cuda/utils.h"
#endif

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_sinks.h>

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

  int read_int_from_env(const char* var, const int default_value) {
    const std::string value = read_string_from_env(var);
    if (value.empty())
      return default_value;
    return std::stoi(value);
  }

  static void log_config() {
    if (!spdlog::should_log(spdlog::level::info))
      return;

    spdlog::info("CPU: {} (SSE4.1={}, AVX={}, AVX2={}, NEON={})",
                 cpu::cpu_vendor(),
                 cpu::cpu_supports_sse41(),
                 cpu::cpu_supports_avx(),
                 cpu::cpu_supports_avx2(),
                 cpu::cpu_supports_neon());
    spdlog::info(" - Selected ISA: {}", cpu::isa_to_str(cpu::get_cpu_isa()));
    spdlog::info(" - Use Intel MKL: {}", cpu::mayiuse_mkl());
    spdlog::info(" - SGEMM backend: {}",
                 cpu::gemm_backend_to_str(cpu::get_gemm_backend(ComputeType::FLOAT)));
    spdlog::info(" - GEMM_S16 backend: {}",
                 cpu::gemm_backend_to_str(cpu::get_gemm_backend(ComputeType::INT16)));
    spdlog::info(" - GEMM_S8 backend: {} (u8s8 preferred: {})",
                 cpu::gemm_backend_to_str(cpu::get_gemm_backend(ComputeType::INT8)),
                 cpu::prefer_u8s8s32_gemm());
    spdlog::info(" - Use packed GEMM: {}", cpu::should_pack_gemm_weights());

#ifdef CT2_WITH_CUDA
    for (int i = 0; i < cuda::get_gpu_count(); ++i) {
      const cudaDeviceProp& device_prop = cuda::get_device_properties(i);
      spdlog::info("GPU #{}: {} (CC={}.{})",
                   i, device_prop.name, device_prop.major, device_prop.minor);
      spdlog::info(" - Allow INT8: {} (with Tensor Cores: {})",
                   mayiuse_int8(Device::CUDA, i),
                   cuda::gpu_has_int8_tensor_cores(i));
      spdlog::info(" - Allow FP16: {} (with Tensor Cores: {})",
                   mayiuse_float16(Device::CUDA, i),
                   cuda::gpu_has_fp16_tensor_cores(i));
    }
#endif
  }

  static void set_log_level(int level) {
    if (level < -3 || level > 3)
      throw std::invalid_argument("Invalid log level "
                                  + std::to_string(level)
                                  + " (should be between -3 and 3)");

    spdlog::set_level(static_cast<spdlog::level::level_enum>(6 - (level + 3)));
  }

  // Initialize the global logger on program start.
  static struct LoggerInit {
    LoggerInit() {
      auto logger = spdlog::stderr_logger_mt("ctranslate2");
      logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [thread %t] [%l] %v");
      spdlog::set_default_logger(logger);
      set_log_level(read_int_from_env("CT2_VERBOSE", 0));
      log_config();
    }
  } logger_init;

  bool mayiuse_float16(Device device, int device_index) {
    switch (device) {
    case Device::CUDA: {
#ifdef CT2_WITH_CUDA
      static const bool allow_float16 = read_bool_from_env("CT2_CUDA_ALLOW_FP16");
      return allow_float16 || cuda::gpu_has_fp16_tensor_cores(device_index);
#else
      (void)device_index;
      return false;
#endif
    }
    default:
      return false;
    }
  }

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
    case Device::CUDA:
#ifdef CT2_WITH_CUDA
      return cuda::gpu_supports_int8(device_index);
#else
      (void)device_index;
      return false;
#endif
    case Device::CPU:
      return cpu::has_gemm_backend(ComputeType::INT8);
    default:
      return false;
    }
  }

  dim_t get_preferred_size_multiple(ComputeType compute_type, Device device, int device_index) {
#ifdef CT2_WITH_CUDA
    if (device == Device::CUDA) {
      if (compute_type == ComputeType::FLOAT16 && cuda::gpu_has_fp16_tensor_cores(device_index))
        return 8;
    }
#else
    (void)compute_type;
    (void)device;
    (void)device_index;
#endif
    return 1;
  }

  int get_gpu_count() {
#ifdef CT2_WITH_CUDA
    return cuda::get_gpu_count();
#else
    return 0;
#endif
  }

  static inline size_t get_default_num_threads() {
    constexpr size_t default_num_threads = 4;
    const size_t max_num_threads = std::thread::hardware_concurrency();
    if (max_num_threads == 0)
      return default_num_threads;
    return std::min(default_num_threads, max_num_threads);
  }

  void set_num_threads(size_t num_threads) {
#ifdef _OPENMP
    if (num_threads == 0)
      num_threads = read_int_from_env("OMP_NUM_THREADS", get_default_num_threads());
    omp_set_num_threads(num_threads);
#else
    (void)num_threads;
#endif
#ifdef CT2_WITH_RUY
    if (num_threads == 0)
      num_threads = get_default_num_threads();
    cpu::get_ruy_context()->set_max_num_threads(num_threads);
#endif
  }

  void set_thread_affinity(std::thread& thread, int index) {
#if !defined(__linux__) || defined(_OPENMP)
    (void)thread;
    (void)index;
    throw std::runtime_error("Setting thread affinity is only supported in Linux binaries built "
                             "with -DOPENMP_RUNTIME=NONE");
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(index, &cpuset);
    const int status = pthread_setaffinity_np(thread.native_handle(), sizeof (cpu_set_t), &cpuset);
    if (status != 0) {
      throw std::runtime_error("Error calling pthread_setaffinity_np: "
                               + std::to_string(status));
    }
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
    return split_string(str, std::string(1, delimiter));
  }

  std::vector<std::string> split_string(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> parts;
    parts.reserve(str.size() / 2);
    size_t offset = 0;

    while (offset < str.size()) {
      size_t pos = str.find(delimiter, offset);
      if (pos == std::string::npos)
        pos = str.size();
      const size_t length = pos - offset;
      if (length > 0)
        parts.emplace_back(str.substr(offset, length));
      offset = pos + delimiter.size();
    }

    return parts;
  }

  constexpr unsigned int default_seed = static_cast<unsigned int>(-1);
  static std::atomic<unsigned int> g_seed(default_seed);

  void set_random_seed(const unsigned int seed) {
    g_seed = seed;
  }

  unsigned int get_random_seed() {
    return g_seed == default_seed ? std::random_device{}() : g_seed.load();
  }

  std::mt19937& get_random_generator() {
    static thread_local std::mt19937 generator(get_random_seed());
    return generator;
  }

}
