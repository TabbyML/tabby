#include "ctranslate2/ops/layer_norm.h"

#include <algorithm>
#include <cmath>

#define EPSILON 1e-5

namespace ctranslate2 {
  namespace ops {

    template <typename T>
    static void layer_norm_kernel(int64_t m,
                                  int64_t n,
                                  T eps,
                                  const T* input,
                                  const T* gamma,
                                  const T* beta,
                                  T* output) {
      #pragma omp parallel for
      for (int64_t i = 0; i < m; ++i) {
        const auto offset = i * n;
        const T* x = input + offset;
        T* y = output + offset;
        T mean = 0;  // sum(x)/n
        T rstd = 0;  // 1/sqrt(var(x)) where var(x) = sum((x-mean)^2)/n = sum(x^2)/n - mean^2
        for (int64_t j = 0; j < n; ++j) {
          mean += x[j];
          rstd += x[j] * x[j];
        }
        mean /= n;
        rstd = std::max(rstd / n - mean * mean, static_cast<T>(0));
        rstd = static_cast<T>(1) / std::sqrt(rstd + eps);
        for (int64_t j = 0; j < n; ++j) {
          y[j] = (x[j] - mean) * rstd * gamma[j] + beta[j];
        }
      }
    }

    template <Device D, typename T>
    void LayerNorm::compute(const StorageView& beta,
                            const StorageView& gamma,
                            const StorageView& input,
                            StorageView& output) const {
      auto depth = input.dim(-1);
      auto batch_size = input.size() / depth;
      layer_norm_kernel(batch_size,
                        depth,
                        static_cast<T>(EPSILON),
                        input.data<T>(),
                        gamma.data<T>(),
                        beta.data<T>(),
                        output.data<T>());
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    LayerNorm::compute<Device::CPU, T>(const StorageView& beta,         \
                                       const StorageView& gamma,        \
                                       const StorageView& input,        \
                                       StorageView& output) const;

    DECLARE_IMPL(float)

  }
}
