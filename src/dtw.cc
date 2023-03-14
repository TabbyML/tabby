#include "dtw.h"

#include <algorithm>
#include <limits>

namespace ctranslate2 {

  static std::vector<std::pair<size_t, size_t>> backtrace(StorageView trace) {
    dim_t i = trace.dim(0) - 1;
    dim_t j = trace.dim(1) - 1;

    for (dim_t k = 0; k < trace.dim(1); ++k)
      trace.at<int32_t>({0, k}) = 2;
    for (dim_t k = 0; k < trace.dim(0); ++k)
      trace.at<int32_t>({k, 0}) = 1;

    std::vector<std::pair<size_t, size_t>> result;

    while (i > 0 && j > 0) {
      result.emplace_back(i - 1, j - 1);

      const auto t = trace.at<int32_t>({i, j});

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

  std::vector<std::pair<size_t, size_t>> negative_dtw(const StorageView& x) {
    const dim_t n = x.dim(0);
    const dim_t m = x.dim(1);

    StorageView cost({n + 1, m + 1}, std::numeric_limits<float>::infinity());
    StorageView trace({n + 1, m + 1}, int32_t(-1));

    cost.at<float>({0, 0}) = 0;

    for (dim_t j = 1; j < m + 1; ++j) {
      for (dim_t i = 1; i < n + 1; ++i) {
        const float c0 = cost.at<float>({i - 1, j - 1});
        const float c1 = cost.at<float>({i - 1, j});
        const float c2 = cost.at<float>({i, j - 1});

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

        cost.at<float>({i, j}) = -x.at<float>({i - 1, j - 1}) + c;
        trace.at<int32_t>({i, j}) = t;
      }
    }

    return backtrace(std::move(trace));
  }

}
