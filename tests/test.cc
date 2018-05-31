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
static void assert_vector_eq(const std::vector<T>& got, const std::vector<T>& expected) {
  ASSERT_EQ(got.size(), expected.size());
  for (size_t i = 0; i < got.size(); ++i) {
    ASSERT_EQ(got[i], expected[i]) << "Value mismatch for dimension " << i;
  }
}

static void expect_storage_eq(const StorageView& got, const StorageView& expected) {
  ASSERT_EQ(got.dtype(), expected.dtype());
  assert_vector_eq(got.shape(), expected.shape());
  TYPE_DISPATCH(got.dtype(), expect_array_eq(got.data<T>(), expected.data<T>(), got.size()));
}

TEST(OpTest, AddFloat) {
  StorageView a({4}, std::vector<float>{1, 2, 3, 4});
  StorageView b({4}, std::vector<float>{2, 3, 4, 5});
  StorageView expected({4}, std::vector<float>{3, 5, 7, 9});
  StorageView c;
  ops::Add()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST(OpTest, ConcatBatch) {
  StorageView a({2, 2}, 1.f);
  StorageView b({1, 2}, 2.f);
  StorageView c({3, 2}, std::vector<float>{1, 1, 1, 1, 2, 2});
  StorageView x;
  ops::Concat(0)({&a, &b}, x);
  expect_storage_eq(x, c);
}

TEST(OpTest, ConcatTime) {
  StorageView a({2, 2, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4});
  StorageView b({2, 1, 2}, std::vector<float>{5, 5, 6, 6});
  StorageView c({2, 3, 2}, std::vector<float>{1, 1, 2, 2, 5, 5, 3, 3, 4, 4, 6, 6});
  StorageView x;
  ops::Concat(1)({&a, &b}, x);
  expect_storage_eq(x, c);
}

TEST(OpTest, ConcatDepth) {
  StorageView a({2, 1}, 1.f);
  StorageView b({2, 2}, 2.f);
  StorageView c({2, 3}, std::vector<float>{1, 2, 2, 1, 2, 2});
  StorageView x;
  ops::Concat(-1)({&a, &b}, x);
  expect_storage_eq(x, c);
}

TEST(OpTest, Transpose1D) {
  StorageView x({4}, std::vector<float>{1, 2, 3, 4});
  StorageView y;
  ops::Transpose()(x, y);
  expect_storage_eq(y, x);
}

TEST(OpTest, Transpose2D) {
  StorageView x({4, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
  StorageView expected({2, 4}, std::vector<float>{1, 3, 5, 7, 2, 4, 6, 8});
  StorageView y;
  ops::Transpose()(x, y);
  expect_storage_eq(y, expected);
  y.release();
  ops::Transpose({1, 0})(x, y);
  expect_storage_eq(y, expected);
  y.release();
  ops::Transpose({0, 1})(x, y);
  expect_storage_eq(y, x);
}

TEST(OpTest, Transpose2DInt16) {
  StorageView x({4, 2}, std::vector<int16_t>{1, 2, 3, 4, 5, 6, 7, 8});
  StorageView expected({2, 4}, std::vector<int16_t>{1, 3, 5, 7, 2, 4, 6, 8});
  StorageView y(x.dtype());
  ops::Transpose()(x, y);
  expect_storage_eq(y, expected);
}

TEST(OpTest, Transpose3D) {
  StorageView x({3, 2, 3},
                std::vector<float>{1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 2, 3, 7, 8, 9, 1, 2, 3});
  StorageView expected({2, 3, 3},
                       std::vector<float>{1, 4, 7, 2, 5, 8, 3, 6, 9, 1, 1, 1, 2, 2, 2, 3, 3, 3});
  StorageView y(x.dtype());
  ops::Transpose({1, 2, 0})(x, y);
  expect_storage_eq(y, expected);
}

TEST(OpTest, Squeeze) {
  StorageView x({2, 1, 3}, DataType::DT_FLOAT);
  StorageView y;
  ops::Squeeze({1})(x, y);
  assert_vector_eq(y.shape(), {2, 3});
  EXPECT_EQ(y.data<float>(), x.data<float>());
  y.release();
  EXPECT_THROW(ops::Squeeze({0})(x, y), std::invalid_argument);
}

TEST(OpTest, Unsqueeze) {
  StorageView x({2, 3}, DataType::DT_FLOAT);
  StorageView y;
  ops::Unsqueeze({1})(x, y);
  assert_vector_eq(y.shape(), {2, 1, 3});
  EXPECT_EQ(y.data<float>(), x.data<float>());
  StorageView z;
  ops::Unsqueeze({0})(y, z);
  assert_vector_eq(z.shape(), {1, 2, 1, 3});
  EXPECT_EQ(z.data<float>(), y.data<float>());
}

TEST(OpTest, Gather) {
  StorageView data({4, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4});
  StorageView ids({2}, std::vector<int32_t>{1, 3});
  StorageView expected({2, 2}, std::vector<float>{2, 2, 4, 4});
  StorageView output;
  ops::Gather(0)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST(OpTest, Gemm) {
  StorageView a({4, 4}, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
  StorageView b(a);
  StorageView c({4, 4}, 2.f);
  StorageView y;
  StorageView expected({4, 4}, std::vector<float>{3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3});
  ops::Gemm op(1.0, 1.0, false, false, false);
  op(a, b, c, y);
  expect_storage_eq(y, expected);
};

TEST(OpTest, GemmInt16) {
  StorageView a({64, 64}, static_cast<int16_t>(1));
  StorageView b(a);
  StorageView c({64, 64}, static_cast<int32_t>(2));
  StorageView y(c.dtype());
  StorageView expected({64, 64}, static_cast<int32_t>(130));
  ops::Gemm op(2.0, 1.0, false, false, true);
  op(a, b, c, y);
  expect_storage_eq(y, expected);
};

TEST(OpTest, Quantize) {
  const float scale = 100;
  const float shift = 5;
  StorageView input({4}, std::vector<float>{0.1f, -0.5f, 2.0f, 0.0f});
  StorageView expected({4}, std::vector<int16_t>{15, -45, 205, 5});
  StorageView output(expected.dtype());
  StorageView reverse(input.dtype());
  ops::Quantize quantize_op(scale, shift);
  ops::Unquantize unquantize_op(scale, shift);
  quantize_op(input, output);
  expect_storage_eq(output, expected);
  unquantize_op(output, reverse);
  expect_storage_eq(reverse, input);
}

TEST(OpTest, TopK) {
  const int k = 3;
  StorageView input({2, 6}, std::vector<float>{0.1, -0.5, 2.0, 0.0, 0.2, 0.6, 1.0, 1.1, 0.2, 0.3, -0.2, 0.0});
  StorageView expected_values({2, 3}, std::vector<float>{2.0, 0.6, 0.2, 1.1, 1.0, 0.3});
  StorageView expected_indices({2, 3}, std::vector<int32_t>{2, 5, 4, 1, 0, 3});
  StorageView values(expected_values.dtype());
  StorageView indices(expected_indices.dtype());
  ops::TopK op(k);
  op(input, values, indices);
  expect_storage_eq(values, expected_values);
  expect_storage_eq(indices, expected_indices);
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
