#include "ctranslate2/utils.h"

#ifdef _OPENMP
#  include <omp.h>
#endif

#ifdef CT2_WITH_CUDA
#  include "./cuda/utils.h"
#endif

#include <spdlog/spdlog.h>

#include "ctranslate2/devices.h"
#include "ctranslate2/logging.h"

#include "cpu/backend.h"
#include "cpu/cpu_info.h"
#include "cpu/cpu_isa.h"
#include "cpu/parallel.h"
#include "env.h"

namespace ctranslate2 {

  bool string_to_bool(const std::string& str) {
    return str == "1" || str == "true" || str == "TRUE";
  }

  void log_system_config() {
    init_logger();

    if (!spdlog::should_log(spdlog::level::info))
      return;

#if defined(CT2_X86_BUILD)
    spdlog::info("CPU: {} (SSE4.1={}, AVX={}, AVX2={}, AVX512={})",
                 cpu::cpu_vendor(),
                 cpu::cpu_supports_sse41(),
                 cpu::cpu_supports_avx(),
                 cpu::cpu_supports_avx2(),
                 cpu::cpu_supports_avx512());
#elif defined(CT2_ARM64_BUILD)
    spdlog::info("CPU: {} (NEON={})",
                 cpu::cpu_vendor(),
                 cpu::cpu_supports_neon());
#endif
    spdlog::info(" - Selected ISA: {}", cpu::isa_to_str(cpu::get_cpu_isa()));
    spdlog::info(" - Use Intel MKL: {}", cpu::mayiuse_mkl());
    spdlog::info(" - SGEMM backend: {} (packed: {})",
                 cpu::gemm_backend_to_str(cpu::get_gemm_backend(ComputeType::FLOAT32)),
                 cpu::pack_gemm_weights(ComputeType::FLOAT32));
    spdlog::info(" - GEMM_S16 backend: {} (packed: {})",
                 cpu::gemm_backend_to_str(cpu::get_gemm_backend(ComputeType::INT16)),
                 cpu::pack_gemm_weights(ComputeType::INT16));
    spdlog::info(" - GEMM_S8 backend: {} (packed: {}, u8s8 preferred: {})",
                 cpu::gemm_backend_to_str(cpu::get_gemm_backend(ComputeType::INT8)),
                 cpu::pack_gemm_weights(ComputeType::INT8),
                 cpu::prefer_u8s8s32_gemm());

#ifdef CT2_WITH_CUDA
    for (int i = 0; i < cuda::get_gpu_count(); ++i) {
      const cudaDeviceProp& device_prop = cuda::get_device_properties(i);
      spdlog::info("GPU #{}: {} (CC={}.{})",
                   i, device_prop.name, device_prop.major, device_prop.minor);
      spdlog::info(" - Allow INT8: {}", mayiuse_int8(Device::CUDA, i));
      spdlog::info(" - Allow FP16: {} (with Tensor Cores: {})",
                   mayiuse_float16(Device::CUDA, i),
                   cuda::gpu_has_fp16_tensor_cores(i));
      spdlog::info(" - Allow BF16: {}", mayiuse_bfloat16(Device::CUDA, i));
    }
#endif
  }

  int get_gpu_count() {
    return get_device_count(Device::CUDA);
  }

  static inline size_t get_default_num_threads() {
    constexpr size_t default_num_threads = 4;
    const size_t max_num_threads = std::thread::hardware_concurrency();
    if (max_num_threads == 0)
      return default_num_threads;
    return std::min(default_num_threads, max_num_threads);
  }

  void set_num_threads(size_t num_threads) {
    if (num_threads == 0)
      num_threads = read_int_from_env("OMP_NUM_THREADS", get_default_num_threads());

#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#else
    cpu::set_num_threads(num_threads);
#endif

#ifdef CT2_WITH_RUY
    cpu::get_ruy_context()->set_max_num_threads(num_threads);
#endif
  }

  std::istream& getline(std::istream& input, std::string& str, bool remove_carriage_return) {
    std::getline(input, str);

    if (remove_carriage_return && !str.empty() && str.back() == '\r')
      str.pop_back();

    return input;
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

  std::string join_string(const std::vector<std::string>& tokens, const std::string& separator) {
    std::string text;
    for (const auto& token : tokens) {
      if (!text.empty())
        text += separator;
      text += token;
    }
    return text;
  }

  std::vector<std::string> split_tokens(const std::string& text) {
    return split_string(text, " ");
  }

  std::string join_tokens(const std::vector<std::string>& tokens) {
    return join_string(tokens, " ");
  }

  std::vector<std::vector<std::vector<std::string>>>
  extract_features(std::vector<std::vector<std::string>> batch,
                   size_t num_features,
                   const std::string& features_separator) {
    std::vector<std::vector<std::vector<std::string>>> features;
    features.resize(num_features);

    if (num_features == 1) {
      features[0] = std::move(batch);
      return features;
    }

    for (const auto& tokens : batch) {
      for (auto& stream : features) {
        stream.emplace_back();
        stream.back().reserve(tokens.size());
      }

      for (const auto& token : tokens) {
        auto fields = split_string(token, features_separator);
        if (fields.size() != num_features)
          throw std::invalid_argument("Expected " + std::to_string(num_features)
                                      + " input features, but token '" + token
                                      + "' has " + std::to_string(fields.size())
                                      + " features");

        for (size_t i = 0; i < fields.size(); ++i)
          features[i].back().emplace_back(std::move(fields[i]));
      }
    }

    return features;
  }
}
