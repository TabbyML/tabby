#include <gtest/gtest.h>

#include "ops.h"

using namespace onmt;

template <typename T>
static void expect_array_eq(const T* x, const T* y, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(x[i], y[i]) << "Value mismatch at index " << i;
  }
}

template <typename T>
static void assert_shape_eq(const StorageView<T>& s, const std::vector<size_t>& shape) {
  ASSERT_EQ(s.rank(), shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    ASSERT_EQ(s.dim(i), shape[i]) << "Value mismatch for dimension " << i;
  }
}

TEST(OpTest, ConcatBatch) {
  StorageView<float> a({2, 2}, 1);
  StorageView<float> b({1, 2}, 2);
  StorageView<float> c;

  std::vector<float> y = { 1, 1,
                           1, 1,
                           2, 2 };

  ops::Concat(0)({&a, &b}, c);

  assert_shape_eq(c, {3, 2});
  expect_array_eq(c.data(), y.data(), y.size());
}

TEST(OpTest, ConcatTime) {
  std::vector<float> a_flat = { 1, 1,
                                2, 2,
                                3, 3,
                                4, 4 };
  std::vector<float> b_flat = { 5, 5,
                                6, 6 };
  std::vector<float> y = { 1, 1,
                           2, 2,
                           5, 5,
                           3, 3,
                           4, 4,
                           6, 6 };

  StorageView<float> a(a_flat.data(), {2, 2, 2});
  StorageView<float> b(b_flat.data(), {2, 1, 2});
  StorageView<float> c;

  ops::Concat(1)({&a, &b}, c);

  assert_shape_eq(c, {2, 3, 2});
  expect_array_eq(c.data(), y.data(), y.size());
}

TEST(OpTest, ConcatDepth) {
  StorageView<float> a({2, 1}, 1);
  StorageView<float> b({2, 2}, 2);
  StorageView<float> c;

  std::vector<float> y = { 1, 2, 2,
                           1, 2, 2 };

  ops::Concat(-1)({&a, &b}, c);

  assert_shape_eq(c, {2, 3});
  expect_array_eq(c.data(), y.data(), y.size());
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
