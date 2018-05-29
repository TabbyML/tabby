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

template <typename T>
static void expect_storage_eq(const StorageView<T>& a, const StorageView<T>& b) {
  assert_shape_eq(b, a.shape());
  expect_array_eq(a.data(), b.data(), a.size());
}


TEST(OpTest, ConcatBatch) {
  StorageView<float> a({2, 2}, 1);
  StorageView<float> b({1, 2}, 2);
  StorageView<float> c({3, 2}, {1, 1, 1, 1, 2, 2});
  StorageView<float> x;
  ops::Concat(0)({&a, &b}, x);
  expect_storage_eq(x, c);
}

TEST(OpTest, ConcatTime) {
  StorageView<float> a({2, 2, 2}, {1, 1, 2, 2, 3, 3, 4, 4});
  StorageView<float> b({2, 1, 2}, {5, 5, 6, 6});
  StorageView<float> c({2, 3, 2}, { 1, 1, 2, 2, 5, 5, 3, 3, 4, 4, 6, 6 });
  StorageView<float> x;
  ops::Concat(1)({&a, &b}, x);
  expect_storage_eq(x, c);
}

TEST(OpTest, ConcatDepth) {
  StorageView<float> a({2, 1}, 1);
  StorageView<float> b({2, 2}, 2);
  StorageView<float> c({2, 3}, {1, 2, 2, 1, 2, 2});
  StorageView<float> x;
  ops::Concat(-1)({&a, &b}, x);
  expect_storage_eq(x, c);
}

TEST(OpTest, Gather) {
  StorageView<float> data({4, 2}, {1, 1, 2, 2, 3, 3, 4, 4});
  StorageView<size_t> ids({2}, {1, 3});
  StorageView<float> expected({2, 2}, {2, 2, 4, 4});
  StorageView<float> output;
  ops::Gather(0)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST(OpTest, Gemm) {
  StorageView<float> a({4, 4}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
  StorageView<float> b(a);
  StorageView<float> c({4, 4}, 2);
  StorageView<float> y;
  StorageView<float> expected({4, 4}, {3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3});
  ops::Gemm op(1.0, 1.0, false, false, false);
  op(a, b, &c, y);
  expect_storage_eq(y, expected);
};

TEST(OpTest, GemmInt16) {
  StorageView<int16_t> a({64, 64}, 1);
  StorageView<int16_t> b(a);
  StorageView<int32_t> c({64, 64}, 2);
  StorageView<int32_t> y;
  StorageView<int32_t> expected({64, 64}, 130);
  ops::Gemm op(2.0, 1.0, false, false, true);
  op(a, b, &c, y);
  expect_storage_eq(y, expected);
};

TEST(OpTest, Quantize) {
  const float scale = 100;
  const float shift = 5;
  StorageView<float> input({4}, {0.1f, -0.5f, 2.0f, 0.0f});
  StorageView<int16_t> expected({4}, {15, -45, 205, 5});
  StorageView<int16_t> output;
  StorageView<float> reverse;
  ops::Quantize quantize_op(scale, shift);
  ops::Unquantize unquantize_op(scale, shift);
  quantize_op(input, output);
  expect_storage_eq(output, expected);
  unquantize_op(output, reverse);
  expect_storage_eq(reverse, input);
}

TEST(OpTest, TopK) {
  const int k = 3;
  StorageView<float> input({2, 6}, {0.1, -0.5, 2.0, 0.0, 0.2, 0.6, 1.0, 1.1, 0.2, 0.3, -0.2, 0.0});
  StorageView<float> expected_values({2, 3}, {2.0, 0.6, 0.2, 1.1, 1.0, 0.3});
  StorageView<size_t> expected_indices({2, 3}, {2, 5, 4, 1, 0, 3});
  StorageView<float> values;
  StorageView<size_t> indices;
  ops::TopK op(k);
  op(input, values, indices);
  expect_storage_eq(values, expected_values);
  expect_storage_eq(indices, expected_indices);
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
