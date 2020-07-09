#include "test_utils.h"
#include "ctranslate2/storage_view.h"

TEST(StorageViewTest, Swap) {
  StorageView a({4}, std::vector<float>{1, 2, 3, 4});
  StorageView b({2, 2}, std::vector<float>{2, 3, 4, 5});
  StorageView expected_a(b);
  StorageView expected_b(a);
  const auto* a_data = a.data<float>();
  const auto* b_data = b.data<float>();
  swap(a, b);
  EXPECT_EQ(a.data<float>(), b_data);
  EXPECT_EQ(b.data<float>(), a_data);
  expect_storage_eq(a, expected_a);
  expect_storage_eq(b, expected_b);
}

TEST(StorageViewTest, ZeroDim) {
  StorageView a({2, 0, 2});
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.rank(), 3);
  EXPECT_EQ(a.dim(0), 2);
  EXPECT_EQ(a.dim(1), 0);
  EXPECT_EQ(a.dim(2), 2);
}

TEST(StorageViewTest, BoolOperator) {
  StorageView a;
  EXPECT_FALSE(bool(a));
  a.resize({4});
  EXPECT_TRUE(bool(a));
}

TEST(StorageViewTest, Reshape) {
  StorageView a(Shape{16});
  assert_vector_eq(a.shape(), Shape{16});
  a.reshape({4, 4});
  assert_vector_eq(a.shape(), Shape{4, 4});
  a.reshape({2, -1});
  assert_vector_eq(a.shape(), Shape{2, 8});
  a.reshape({-1, 1});
  assert_vector_eq(a.shape(), Shape{16, 1});
  a.reshape({2, -1, 2});
  assert_vector_eq(a.shape(), Shape{2, 4, 2});
  a.reshape({-1});
  assert_vector_eq(a.shape(), Shape{16});
}

class StorageViewDeviceTest : public ::testing::TestWithParam<Device> {
};

TEST_P(StorageViewDeviceTest, HalfConversion) {
  const Device device = GetParam();
  const StorageView a({4}, std::vector<float>{1, 2, 3, 4}, device);
  EXPECT_EQ(a.reserved_memory(), 4 * 4);
  const StorageView b = a.to_float16();
  EXPECT_EQ(b.dtype(), DataType::FLOAT16);
  EXPECT_EQ(b.reserved_memory(), 4 * 2);
  expect_storage_eq(b.to_float(), a);
}

INSTANTIATE_TEST_CASE_P(CPU, StorageViewDeviceTest, ::testing::Values(Device::CPU));
#ifdef CT2_WITH_CUDA
INSTANTIATE_TEST_CASE_P(CUDA, StorageViewDeviceTest, ::testing::Values(Device::CUDA));
#endif
