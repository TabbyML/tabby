#include "test_utils.h"
#include "ctranslate2/primitives.h"
#include "dispatch.h"

class PrimitiveTest : public ::testing::TestWithParam<Device> {
};

TEST_P(PrimitiveTest, StridedFill) {
  const Device device = GetParam();
  StorageView x({3, 2}, float(0), device);
  StorageView expected({3, 2}, std::vector<float>{1, 0, 1, 0, 1, 0}, device);
  DEVICE_DISPATCH(device, primitives<D>::strided_fill(x.data<float>(), 1.f, 2, 3));
  expect_storage_eq(x, expected);
}

TEST_P(PrimitiveTest, PenalizeTokens) {
  const Device device = GetParam();
  const float penalty = 1.2f;
  StorageView ids({2}, std::vector<int32_t>{2, 1}, device);
  StorageView scores({2, 4}, std::vector<float>{0.6, 0.2, -1.2, 0.1, 0.3, 0.5, -1.3, 0.2});
  StorageView expected = scores;
  expected.at<float>({0, 2}) *= penalty;
  expected.at<float>({1, 1}) /= penalty;
  scores = scores.to(device);
  DEVICE_DISPATCH(device, primitives<D>::penalize_tokens(scores.data<float>(),
                                                         ids.data<int32_t>(),
                                                         penalty, 2, 4));
  expect_storage_eq(scores, expected);
}

INSTANTIATE_TEST_CASE_P(CPU, PrimitiveTest, ::testing::Values(Device::CPU));
#ifdef CT2_WITH_CUDA
INSTANTIATE_TEST_CASE_P(CUDA, PrimitiveTest, ::testing::Values(Device::CUDA));
#endif
