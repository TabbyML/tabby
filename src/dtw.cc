#include "dtw.h"

#include <algorithm>
#include <limits>

namespace ctranslate2 {

  static std::vector<std::pair<dim_t, dim_t>> backtrace(std::vector<std::vector<int>> trace,
                                                        dim_t i,
                                                        dim_t j) {
    for (dim_t k = 0; k <= j; ++k)
      trace[0][k] = 2;
    for (dim_t k = 0; k <= i; ++k)
      trace[k][0] = 1;

    std::vector<std::pair<dim_t, dim_t>> result;

    while (i > 0 || j > 0) {
      result.emplace_back(i - 1, j - 1);

      const int t = trace[i][j];

      if (t == 0) {
        --i;
        --j;
      } else if (t == 1) {
        --i;
      } else if (t == 2) {
        --j;
      } else {
        throw std::runtime_error("Unexpected trace[i, j]");
      }
    }

    std::reverse(result.begin(), result.end());

    return result;
  }

  std::vector<std::pair<dim_t, dim_t>> negative_dtw(const StorageView& x) {
    constexpr float inf = std::numeric_limits<float>::infinity();
    const dim_t n = x.dim(0);
    const dim_t m = x.dim(1);

    std::vector<std::vector<float>> cost(n + 1, std::vector<float>(m + 1, inf));
    std::vector<std::vector<int>> trace(n + 1, std::vector<int>(m + 1, -1));

    cost[0][0] = 0;

    const auto* x_data = x.data<float>();

    for (dim_t j = 1; j < m + 1; ++j) {
      for (dim_t i = 1; i < n + 1; ++i) {
        const float c0 = cost[i - 1][j - 1];
        const float c1 = cost[i - 1][j];
        const float c2 = cost[i][j - 1];

        float c = 0;
        int t = 0;

        if (c0 < c1 && c0 < c2) {
          c = c0;
          t = 0;
        } else if (c1 < c0 && c1 < c2) {
          c = c1;
          t = 1;
        } else {
          c = c2;
          t = 2;
        }

        const float v = x_data[(i - 1) * x.dim(1) + (j - 1)];

        cost[i][j] = -v + c;
        trace[i][j] = t;
      }
    }

    return backtrace(std::move(trace), n, m);
  }

}
