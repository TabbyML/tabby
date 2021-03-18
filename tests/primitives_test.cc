#include "test_utils.h"
#include "ctranslate2/primitives.h"

TEST(PrimitiveTest, StridedFillCPU) {
  StorageView x({3, 2}, float(0));
  StorageView expected({3, 2}, std::vector<float>{1, 0, 1, 0, 1, 0});
  primitives<Device::CPU>::strided_fill(x.data<float>(), 1.f, 2, 3);
  expect_storage_eq(x, expected);
}
