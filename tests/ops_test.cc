#include "test_utils.h"
#include "opennmt/ops/ops.h"

TEST(OpTest, AddFloat) {
  StorageView a({4}, std::vector<float>{1, 2, 3, 4});
  StorageView b({4}, std::vector<float>{2, 3, 4, 5});
  StorageView expected({4}, std::vector<float>{3, 5, 7, 9});
  StorageView c;
  ops::Add()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST(OpTest, ConcatSplitBatch) {
  StorageView a({2, 2}, std::vector<float>{1, 2, 3, 4});
  StorageView b({1, 2}, std::vector<float>{5, 6});
  StorageView c({3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6});
  StorageView x;
  ops::Concat(0)({&a, &b}, x);
  expect_storage_eq(x, c);
  StorageView y, z;
  std::vector<StorageView*> out{&y, &z};
  ops::Split(0, {2, 1})(c, out);
  expect_storage_eq(y, a);
  expect_storage_eq(z, b);
}

TEST(OpTest, ConcatSplitTime) {
  StorageView a({2, 2, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4});
  StorageView b({2, 1, 2}, std::vector<float>{5, 5, 6, 6});
  StorageView c({2, 3, 2}, std::vector<float>{1, 1, 2, 2, 5, 5, 3, 3, 4, 4, 6, 6});
  StorageView x;
  ops::Concat(1)({&a, &b}, x);
  expect_storage_eq(x, c);
  StorageView y, z;
  std::vector<StorageView*> out{&y, &z};
  ops::Split(1, {2, 1})(c, out);
  expect_storage_eq(y, a);
  expect_storage_eq(z, b);
}

TEST(OpTest, ConcatSplitDepth) {
  StorageView a({2, 1}, std::vector<float>{1, 4});
  StorageView b({2, 2}, std::vector<float>{2, 3, 5, 6});
  StorageView c({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  StorageView x;
  ops::Concat(-1)({&a, &b}, x);
  expect_storage_eq(x, c);
  StorageView y, z;
  std::vector<StorageView*> out{&y, &z};
  ops::Split(-1, {1, 2})(c, out);
  expect_storage_eq(y, a);
  expect_storage_eq(z, b);
}

TEST(OpTest, ConcatSplitDepth3) {
  StorageView a({2, 2}, std::vector<float>{1, 2, 6, 7});
  StorageView b({2, 1}, std::vector<float>{3, 8});
  StorageView c({2, 2}, std::vector<float>{4, 5, 9, 10});
  StorageView d({2, 5}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  StorageView x;
  ops::Concat(-1)({&a, &b, &c}, x);
  expect_storage_eq(x, d);
  StorageView y, z, w;
  std::vector<StorageView*> out{&y, &z, &w};
  ops::Split(-1, {2, 1, 2})(d, out);
  expect_storage_eq(y, a);
  expect_storage_eq(z, b);
  expect_storage_eq(w, c);
}

TEST(OpTest, ConcatSplitDepthEqualParts) {
  StorageView a({2, 2}, std::vector<float>{1, 2, 5, 6});
  StorageView b({2, 2}, std::vector<float>{3, 4, 7, 8});
  StorageView c({2, 4}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
  StorageView x;
  ops::Concat(-1)({&a, &b}, x);
  expect_storage_eq(x, c);
  StorageView y, z;
  std::vector<StorageView*> out{&y, &z};
  ops::Split(-1)(c, out);
  expect_storage_eq(y, a);
  expect_storage_eq(z, b);
}

TEST(OpTest, TileFirstDim) {
  StorageView input({2, 2}, std::vector<float>{1, 2, 3, 4});
  StorageView repeats({2}, std::vector<int32_t>{2, 1});
  StorageView expected_output({4, 2}, std::vector<float>{1, 2, 3, 4, 1, 2, 3, 4});
  StorageView output;
  ops::Tile()(input, repeats, output);
  expect_storage_eq(output, expected_output);
}

TEST(OpTest, TileLastDim) {
  StorageView input({2, 2}, std::vector<float>{1, 2, 3, 4});
  StorageView repeats({2}, std::vector<int32_t>{1, 2});
  StorageView expected_output({2, 4}, std::vector<float>{1, 2, 1, 2, 3, 4, 3, 4});
  StorageView output;
  ops::Tile()(input, repeats, output);
  expect_storage_eq(output, expected_output);
}

TEST(OpTest, TileAll2D) {
  StorageView input({2, 2}, std::vector<float>{1, 2, 3, 4});
  StorageView repeats({2}, std::vector<int32_t>{2, 2});
  StorageView expected_output({4, 4}, std::vector<float>{1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4});
  StorageView output;
  ops::Tile()(input, repeats, output);
  expect_storage_eq(output, expected_output);
}

TEST(OpTest, TileMiddleDim) {
  StorageView input({2, 1, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
  StorageView repeats({3}, std::vector<int32_t>{1, 3, 1});
  StorageView expected_output({2, 3, 3}, std::vector<float>{1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6});
  StorageView output;
  ops::Tile()(input, repeats, output);
  expect_storage_eq(output, expected_output);
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

TEST(OpTest, Transpose3DReverse) {
  StorageView x({3, 2, 3},
                std::vector<float>{1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 2, 3, 7, 8, 9, 1, 2, 3});
  StorageView expected({3, 2, 3},
                       std::vector<float>{1, 4, 7, 1, 1, 1, 2, 5, 8, 2, 2, 2, 3, 6, 9, 3, 3, 3});
  StorageView y(x.dtype());
  ops::Transpose()(x, y);
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

TEST(OpTest, GatherData1D) {
  StorageView data({4}, std::vector<float>{1, 2, 3, 4});
  StorageView ids({2}, std::vector<int32_t>{1, 3});
  StorageView expected({2}, std::vector<float>{2, 4});
  StorageView output;
  ops::Gather(0)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST(OpTest, GatherData1DIndex2D) {
  StorageView data({4}, std::vector<float>{1, 2, 3, 4});
  StorageView ids({2, 3}, std::vector<int32_t>{1, 3, 1, 1, 2, 0});
  StorageView expected({2, 3}, std::vector<float>{2, 4, 2, 2, 3, 1});
  StorageView output;
  ops::Gather(0)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST(OpTest, GatherData2D) {
  StorageView data({4, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4});
  StorageView ids({2}, std::vector<int32_t>{1, 3});
  StorageView expected({2, 2}, std::vector<float>{2, 2, 4, 4});
  StorageView output;
  ops::Gather(0)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST(OpTest, GatherData3D) {
  StorageView data({2, 3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  StorageView ids({2}, std::vector<int32_t>{1, 1});
  StorageView expected({2, 3, 2}, std::vector<float>{7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12});
  StorageView output;
  ops::Gather(0)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST(OpTest, GatherData2DIndex2D) {
  StorageView data({4, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4});
  StorageView ids({2, 3}, std::vector<int32_t>{1, 3, 3, 2, 1, 0});
  StorageView expected({2, 3, 2}, std::vector<float>{2, 2, 4, 4, 4, 4, 3, 3, 2, 2, 1, 1});
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
