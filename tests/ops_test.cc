#include <algorithm>
#include "test_utils.h"
#include "ctranslate2/layers/attention.h"
#include "ctranslate2/ops/ops.h"

TEST(OpTest, Transpose1D) {
  StorageView x({4}, std::vector<float>{1, 2, 3, 4});
  StorageView y;
  ops::Transpose()(x, y);
  expect_storage_eq(y, x);
}

TEST(OpTest, Squeeze) {
  StorageView x({2, 1, 3}, DataType::FLOAT32);
  StorageView y;
  ops::Squeeze({1})(x, y);
  assert_vector_eq(y.shape(), {2, 3});
  EXPECT_EQ(y.data<float>(), x.data<float>());
  y.release();
  EXPECT_THROW(ops::Squeeze({0})(x, y), std::invalid_argument);
}

TEST(OpTest, Unsqueeze) {
  StorageView x({2, 3}, DataType::FLOAT32);
  StorageView y;
  ops::Unsqueeze({1})(x, y);
  assert_vector_eq(y.shape(), {2, 1, 3});
  EXPECT_EQ(y.data<float>(), x.data<float>());
  StorageView z;
  ops::Unsqueeze({0})(y, z);
  assert_vector_eq(z.shape(), {1, 2, 1, 3});
  EXPECT_EQ(z.data<float>(), y.data<float>());
}

TEST(OpTest, SplitNoCopyInvalidArgument) {
  ASSERT_RAISES(ops::Split(1, /*no_copy=*/true), std::invalid_argument);
}

TEST(OpDeviceTest, SplitInvalidSize) {
  StorageView x({4, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
  StorageView a, b;
  ASSERT_RAISES(ops::Split(0, {3, 2})(x, a, b), std::invalid_argument);
}

TEST(OpDeviceTest, SplitInvalidNumSplits) {
  StorageView x({4, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
  StorageView a, b, c;
  ASSERT_RAISES(ops::Split(0, {3, 1})(x, a, b, c), std::invalid_argument);
}

TEST(OpDeviceTest, SplitInvalidNumOutputs) {
  StorageView x({4, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
  StorageView a, b, c;
  ASSERT_RAISES(ops::Split(0)(x, a, b, c), std::invalid_argument);
}

TEST(OpDeviceTest, GatherInPlaceStrictlyIncreasing) {
  StorageView data({4, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4});
  void* data_ptr = data.buffer();
  StorageView ids({2}, std::vector<int32_t>{1, 2});
  StorageView expected({2, 2}, std::vector<float>{2, 2, 3, 3});
  ops::Gather(0)(data, ids);
  expect_storage_eq(data, expected);
  EXPECT_EQ(data.buffer(), data_ptr);
}

TEST(OpDeviceTest, GatherInPlaceIncreasing) {
  StorageView data({4, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4});
  void* data_ptr = data.buffer();
  StorageView ids({3}, std::vector<int32_t>{0, 0, 1});
  StorageView expected({3, 2}, std::vector<float>{1, 1, 1, 1, 2, 2});
  ops::Gather(0)(data, ids);
  expect_storage_eq(data, expected);
  EXPECT_NE(data.buffer(), data_ptr);
}

TEST(OpDeviceTest, GatherInPlaceDecreasing) {
  StorageView data({4, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4});
  void* data_ptr = data.buffer();
  StorageView ids({2}, std::vector<int32_t>{1, 0});
  StorageView expected({2, 2}, std::vector<float>{2, 2, 1, 1});
  ops::Gather(0)(data, ids);
  expect_storage_eq(data, expected);
  EXPECT_NE(data.buffer(), data_ptr);
}

TEST(OpDeviceTest, GatherInPlaceLarger) {
  StorageView data({4, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4});
  void* data_ptr = data.buffer();
  StorageView ids({5}, std::vector<int32_t>{0, 1, 2, 3, 3});
  StorageView expected({5, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 4, 4});
  ops::Gather(0)(data, ids);
  expect_storage_eq(data, expected);
  EXPECT_NE(data.buffer(), data_ptr);
}

TEST(OpTest, GemmInt16) {
  if (!mayiuse_int16(Device::CPU))
    return;
  StorageView a({64, 64}, static_cast<int16_t>(1));
  StorageView b(a);
  StorageView y({64, 64}, static_cast<int32_t>(2));
  StorageView expected({64, 64}, static_cast<int32_t>(130));
  ops::Gemm op(2.0, 1.0, false, true);
  op(a, b, y);
  expect_storage_eq(y, expected);
};

TEST(OpTest, QuantizeINT16) {
  StorageView scale;
  StorageView input({4}, std::vector<float>{0.1f, -0.5f, 2.0f, 0.0f});
  StorageView expected({4}, std::vector<int16_t>{100, -500, 2000, 0});
  StorageView output(expected.dtype());
  StorageView reverse(input.dtype());
  ops::Quantize()(input, output, scale);
  expect_storage_eq(output, expected);
  ops::Dequantize()(output, scale, reverse);
  expect_storage_eq(reverse, input);
}


class OpDeviceTest : public ::testing::TestWithParam<Device> {
};

class OpDeviceFPTest : public ::testing::TestWithParam<std::pair<Device, DataType>> {
};


TEST_P(OpDeviceTest, Add) {
  Device device = GetParam();
  StorageView a({4}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView b({4}, std::vector<float>{2, 3, 4, 5}, device);
  StorageView expected({4}, std::vector<float>{3, 5, 7, 9}, device);
  StorageView c(a.device());
  ops::Add()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, AddScalar) {
  Device device = GetParam();
  StorageView a({4}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView b(static_cast<float>(3));
  StorageView expected({4}, std::vector<float>{4, 5, 6, 7}, device);
  StorageView c(a.device());
  ops::Add()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, Mul) {
  Device device = GetParam();
  StorageView a({4}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView b({4}, std::vector<float>{2, 3, 4, 5}, device);
  StorageView expected({4}, std::vector<float>{2, 6, 12, 20}, device);
  StorageView c(a.device());
  ops::Mul()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, MulScalar) {
  Device device = GetParam();
  StorageView a({4}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView b(static_cast<float>(3));
  StorageView expected({4}, std::vector<float>{3, 6, 9, 12}, device);
  StorageView c(a.device());
  ops::Mul()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, Sub) {
  Device device = GetParam();
  StorageView a({4}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView b({4}, std::vector<float>{2, 3, 4, 5}, device);
  StorageView expected({4}, std::vector<float>{-1, -1, -1, -1}, device);
  StorageView c(a.device());
  ops::Sub()(a, b, c);
  expect_storage_eq(c, expected);
}

TEST_P(OpDeviceTest, TileFirstDim) {
  Device device = GetParam();
  StorageView input({2, 2}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView expected_output({4, 2}, std::vector<float>{1, 2, 3, 4, 1, 2, 3, 4}, device);
  StorageView output(device);
  ops::Tile(0, 2)(input, output);
  expect_storage_eq(output, expected_output);
}

TEST_P(OpDeviceTest, TileLastDim) {
  Device device = GetParam();
  StorageView input({2, 2}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView expected_output({2, 4}, std::vector<float>{1, 2, 1, 2, 3, 4, 3, 4}, device);
  StorageView output(device);
  ops::Tile(1, 2)(input, output);
  expect_storage_eq(output, expected_output);
}

TEST_P(OpDeviceTest, TileMiddleDim) {
  Device device = GetParam();
  StorageView input({2, 1, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}, device);
  StorageView expected_output({2, 3, 3}, std::vector<float>{1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6}, device);
  StorageView output(device);
  ops::Tile(1, 3)(input, output);
  expect_storage_eq(output, expected_output);
}

TEST_P(OpDeviceTest, ConcatEmpty) {
  Device device = GetParam();
  StorageView a({2, 1, 2}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView b({2, 0, 2}, DataType::FLOAT32, device);
  StorageView x(device);
  ops::Concat(1)({&a, &b}, x);
  expect_storage_eq(x, a);
}

TEST_P(OpDeviceTest, ConcatSplitBatch) {
  Device device = GetParam();
  StorageView a({2, 2}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView b({1, 2}, std::vector<float>{5, 6}, device);
  StorageView c({3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6}, device);
  StorageView x(device);
  ops::Concat(0)({&a, &b}, x);
  expect_storage_eq(x, c);
  StorageView y(device);
  StorageView z(device);
  std::vector<StorageView*> out{&y, &z};
  ops::Split(0, {2, 1})(c, out);
  expect_storage_eq(y, a);
  expect_storage_eq(z, b);
}

TEST_P(OpDeviceTest, ConcatSplitTime) {
  Device device = GetParam();
  StorageView a({2, 2, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4}, device);
  StorageView b({2, 1, 2}, std::vector<float>{5, 5, 6, 6}, device);
  StorageView c({2, 3, 2}, std::vector<float>{1, 1, 2, 2, 5, 5, 3, 3, 4, 4, 6, 6}, device);
  StorageView x(device);
  ops::Concat(1)({&a, &b}, x);
  expect_storage_eq(x, c);
  StorageView y(device);
  StorageView z(device);
  std::vector<StorageView*> out{&y, &z};
  ops::Split(1, {2, 1})(c, out);
  expect_storage_eq(y, a);
  expect_storage_eq(z, b);
}

TEST_P(OpDeviceTest, ConcatSplitDepth) {
  Device device = GetParam();
  StorageView a({2, 1}, std::vector<float>{1, 4}, device);
  StorageView b({2, 2}, std::vector<float>{2, 3, 5, 6}, device);
  StorageView c({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}, device);
  StorageView x(device);
  ops::Concat(-1)({&a, &b}, x);
  expect_storage_eq(x, c);
  StorageView y(device);
  StorageView z(device);
  std::vector<StorageView*> out{&y, &z};
  ops::Split(-1, {1, 2})(c, out);
  expect_storage_eq(y, a);
  expect_storage_eq(z, b);
}

TEST_P(OpDeviceTest, ConcatSplitDepth3) {
  Device device = GetParam();
  StorageView a({2, 2}, std::vector<float>{1, 2, 6, 7}, device);
  StorageView b({2, 1}, std::vector<float>{3, 8}, device);
  StorageView c({2, 2}, std::vector<float>{4, 5, 9, 10}, device);
  StorageView d({2, 5}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, device);
  StorageView x(device);
  ops::Concat(-1)({&a, &b, &c}, x);
  expect_storage_eq(x, d);
  StorageView w(device);
  StorageView y(device);
  StorageView z(device);
  std::vector<StorageView*> out{&w, &y, &z};
  ops::Split(-1, {2, 1, 2})(d, out);
  expect_storage_eq(w, a);
  expect_storage_eq(y, b);
  expect_storage_eq(z, c);
}

TEST_P(OpDeviceTest, ConcatSplitDepthEqualParts) {
  Device device = GetParam();
  StorageView a({2, 2}, std::vector<float>{1, 2, 5, 6}, device);
  StorageView b({2, 2}, std::vector<float>{3, 4, 7, 8}, device);
  StorageView c({2, 4}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, device);
  StorageView x(device);
  ops::Concat(-1)({&a, &b}, x);
  expect_storage_eq(x, c);
  StorageView y(device);
  StorageView z(device);
  std::vector<StorageView*> out{&y, &z};
  ops::Split(-1)(c, out);
  expect_storage_eq(y, a);
  expect_storage_eq(z, b);
}

TEST_P(OpDeviceTest, SplitNoCopy) {
  Device device = GetParam();
  StorageView x({4, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, device);
  StorageView y(device);
  StorageView z(device);
  ops::Split(0, {3, 1}, /*no_copy=*/true)(x, y, z);
  assert_vector_eq(y.shape(), Shape{3, 2});
  assert_vector_eq(z.shape(), Shape{1, 2});
  EXPECT_EQ(y.data<float>(), x.data<float>());
  EXPECT_EQ(z.data<float>(), x.data<float>() + 3 * 2);
}

TEST_P(OpDeviceTest, SplitNoCopyEqualParts) {
  Device device = GetParam();
  StorageView x({4, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, device);
  StorageView y(device);
  StorageView z(device);
  ops::Split(0, /*no_copy=*/true)(x, y, z);
  assert_vector_eq(y.shape(), Shape{2, 2});
  assert_vector_eq(z.shape(), Shape{2, 2});
  EXPECT_EQ(y.data<float>(), x.data<float>());
  EXPECT_EQ(z.data<float>(), x.data<float>() + 4);
}

TEST_P(OpDeviceTest, Mean) {
  const Device device = GetParam();
  const StorageView input({2, 3, 2}, std::vector<float>{
      1, 2, 3, 4, 5, 6,
      7, 8, 9, 10, 11, 12
    }, device);
  StorageView output(device);

  {
    ops::Mean(0)(input, output);
    const StorageView expected({3, 2}, std::vector<float>{4, 5, 6, 7, 8, 9}, device);
    expect_storage_eq(output, expected);
  }

  {
    ops::Mean(1)(input, output);
    const StorageView expected({2, 2}, std::vector<float>{3, 4, 9, 10}, device);
    expect_storage_eq(output, expected);
  }

  {
    ops::Mean(-1)(input, output);
    const StorageView expected({2, 3}, std::vector<float>{1.5, 3.5, 5.5, 7.5, 9.5, 11.5}, device);
    expect_storage_eq(output, expected);
  }
}

TEST_P(OpDeviceTest, GatherData1D) {
  Device device = GetParam();
  StorageView data({4}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView ids({2}, std::vector<int32_t>{1, 3}, device);
  StorageView expected({2}, std::vector<float>{2, 4}, device);
  StorageView output(device);
  ops::Gather(0)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST_P(OpDeviceTest, GatherData1DIndex2D) {
  Device device = GetParam();
  StorageView data({4}, std::vector<float>{1, 2, 3, 4}, device);
  StorageView ids({2, 3}, std::vector<int32_t>{1, 3, 1, 1, 2, 0}, device);
  StorageView expected({2, 3}, std::vector<float>{2, 4, 2, 2, 3, 1}, device);
  StorageView output(device);
  ops::Gather(0)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST_P(OpDeviceTest, GatherData2D) {
  Device device = GetParam();
  StorageView data({4, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4}, device);
  StorageView ids({2}, std::vector<int32_t>{1, 3}, device);
  StorageView expected({2, 2}, std::vector<float>{2, 2, 4, 4}, device);
  StorageView output(device);
  ops::Gather(0)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST_P(OpDeviceTest, GatherData3D) {
  Device device = GetParam();
  StorageView data({2, 3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, device);
  StorageView ids({2}, std::vector<int32_t>{1, 1}, device);
  StorageView expected({2, 3, 2}, std::vector<float>{7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12}, device);
  StorageView output(device);
  ops::Gather(0)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST_P(OpDeviceTest, GatherData2DIndex2D) {
  Device device = GetParam();
  StorageView data({4, 2}, std::vector<float>{1, 1, 2, 2, 3, 3, 4, 4}, device);
  StorageView ids({2, 3}, std::vector<int32_t>{1, 3, 3, 2, 1, 0}, device);
  StorageView expected({2, 3, 2}, std::vector<float>{2, 2, 4, 4, 4, 4, 3, 3, 2, 2, 1, 1}, device);
  StorageView output(device);
  ops::Gather(0)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST_P(OpDeviceTest, GatherInDepthWith1DInput) {
  Device device = GetParam();
  StorageView data({2, 4}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, device);
  StorageView ids({2}, std::vector<int32_t>{1, 3}, device);
  StorageView expected({2}, std::vector<float>{2, 8}, device);
  StorageView output(device);
  ops::Gather(-1, 1)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST_P(OpDeviceTest, GatherInDepthWith2DInput) {
  Device device = GetParam();
  StorageView data({2, 4}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, device);
  StorageView ids({2, 2}, std::vector<int32_t>{1, 2, 0, 3}, device);
  StorageView expected({2, 2}, std::vector<float>{2, 3, 5, 8}, device);
  StorageView output(device);
  ops::Gather(-1, 1)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST_P(OpDeviceTest, GatherInTime) {
  Device device = GetParam();
  StorageView data({2, 3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, device);
  StorageView ids({2, 2}, std::vector<int32_t>{1, 1, 2, 0}, device);
  StorageView expected({2, 2, 2}, std::vector<float>{3, 4, 3, 4, 11, 12, 7, 8}, device);
  StorageView output(device);
  ops::Gather(1, 1)(data, ids, output);
  expect_storage_eq(output, expected);
}

TEST_P(OpDeviceTest, Transpose2D) {
  Device device = GetParam();
  StorageView x({4, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, device);
  StorageView expected({2, 4}, std::vector<float>{1, 3, 5, 7, 2, 4, 6, 8}, device);
  StorageView y(device);
  ops::Transpose()(x, y);
  expect_storage_eq(y, expected);
  y.release();
  ops::Transpose({1, 0})(x, y);
  expect_storage_eq(y, expected);
  y.release();
  ops::Transpose({0, 1})(x, y);
  expect_storage_eq(y, x);
}

TEST_P(OpDeviceTest, Transpose2DInt16) {
  Device device = GetParam();
  StorageView x({4, 2}, std::vector<int16_t>{1, 2, 3, 4, 5, 6, 7, 8}, device);
  StorageView expected({2, 4}, std::vector<int16_t>{1, 3, 5, 7, 2, 4, 6, 8}, device);
  StorageView y(x.dtype(), x.device());
  ops::Transpose()(x, y);
  expect_storage_eq(y, expected);
}

TEST_P(OpDeviceTest, Transpose3D) {
  Device device = GetParam();
  StorageView x({3, 2, 3},
                std::vector<float>{1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 2, 3, 7, 8, 9, 1, 2, 3}, device);
  StorageView expected({2, 3, 3},
                       std::vector<float>{1, 4, 7, 2, 5, 8, 3, 6, 9, 1, 1, 1, 2, 2, 2, 3, 3, 3}, device);
  StorageView y(x.dtype(), x.device());
  ops::Transpose({1, 2, 0})(x, y);
  expect_storage_eq(y, expected);
}

TEST_P(OpDeviceTest, Transpose3DReverse) {
  Device device = GetParam();
  StorageView x({3, 2, 3},
                std::vector<float>{1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 2, 3, 7, 8, 9, 1, 2, 3}, device);
  StorageView expected({3, 2, 3},
                       std::vector<float>{1, 4, 7, 1, 1, 1, 2, 5, 8, 2, 2, 2, 3, 6, 9, 3, 3, 3}, device);
  StorageView y(x.dtype(), x.device());
  ops::Transpose()(x, y);
  expect_storage_eq(y, expected);
}

TEST_P(OpDeviceFPTest, Gemm) {
  const Device device = GetParam().first;
  const DataType dtype = GetParam().second;
  StorageView a(
    {4, 4}, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, device);
  StorageView b(a);
  StorageView y({4, 4}, 2.f, device);
  StorageView expected(
    {4, 4}, std::vector<float>{3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3}, device);
  ops::Gemm op(1.0, 1.0, false, false);
  y = y.to(dtype);
  op(a.to(dtype), b.to(dtype), y);
  expect_storage_eq(y.to_float(), expected);
};

TEST_P(OpDeviceTest, GemmInt8) {
  Device device = GetParam();
  if (!mayiuse_int8(device))
    return;
  StorageView a({3, 8}, std::vector<int8_t>{
      -31, 14, -39, 36, 17, 4, -10, 15,
      -58, 8, 0, -26, -18, -42, -3, -21,
      -27, -63, -51, -4, -37, -63,  2, -4}, device);
  StorageView b({8, 4}, std::vector<int8_t>{
      42, -59, -28, 50,
      56, -17, 14, -57,
      -15, 37, 37, 63,
      -29, 41, -41, 9,
      -47, 38, -20, 27,
      54, 16, -11, -31,
      32, -17, -10, -58,
      45, -17, 58, -44}, device);
  StorageView c(DataType::INT32, device);
  StorageView expected({3, 4}, std::vector<int32_t>{
      -1205, 2249, -1269, -4226,
      -3697, 1272, 2436, -1676,
      -5560, -1767, -668, 6}, device);
  ops::Gemm op(1.0, 0.0, false, false);
  op(a, b, c);
  expect_storage_eq(c, expected);
};

TEST_P(OpDeviceFPTest, TopK) {
  const Device device = GetParam().first;
  const DataType dtype = GetParam().second;
  const int k = 3;
  StorageView input({2, 6}, std::vector<float>{0.1, -0.5, 2.0, 0.0, 0.2, 0.6, 1.0, 1.1, 0.2, 0.3, -0.2, 0.0}, device);
  StorageView expected_values({2, 3}, std::vector<float>{2.0, 0.6, 0.2, 1.1, 1.0, 0.3}, device);
  StorageView expected_indices({2, 3}, std::vector<int32_t>{2, 5, 4, 1, 0, 3}, device);
  StorageView values(dtype, device);
  StorageView indices(expected_indices.dtype(), device);
  ops::TopK op(k);
  op(input.to(dtype), values, indices);
  expect_storage_eq(values.to_float(), expected_values, 1e-3);
  expect_storage_eq(indices, expected_indices);
}

TEST_P(OpDeviceTest, TopKVariableDepth) {
  Device device = GetParam();
  const int k = 3;
  ops::TopK op(k);
  StorageView input({2, 6}, std::vector<float>{0.1, -0.5, 2.0, 0.0, 0.2, 0.6, 1.0, 1.1, 0.2, 0.3, -0.2, 0.0}, device);
  StorageView expected_values({2, 3}, std::vector<float>{2.0, 0.6, 0.2, 1.1, 1.0, 0.3}, device);
  StorageView expected_indices({2, 3}, std::vector<int32_t>{2, 5, 4, 1, 0, 3}, device);
  StorageView values(expected_values.dtype(), device);
  StorageView indices(expected_indices.dtype(), device);
  op(input, values, indices);
  expect_storage_eq(values, expected_values);
  expect_storage_eq(indices, expected_indices);
  StorageView input2({2, 4}, std::vector<float>{0.1, 2.0, 0.2, 0.6, 1.0, 1.1, 0.2, 0.3}, device);
  StorageView expected_values2({2, 3}, std::vector<float>{2.0, 0.6, 0.2, 1.1, 1.0, 0.3}, device);
  StorageView expected_indices2({2, 3}, std::vector<int32_t>{1, 3, 2, 1, 0, 3}, device);
  op(input2, values, indices);
  expect_storage_eq(values, expected_values2);
  expect_storage_eq(indices, expected_indices2);
}

TEST_P(OpDeviceTest, TopKChangeK) {
  const Device device = GetParam();
  const StorageView input({2, 6},
                          std::vector<float>{
                            0.1, -0.5, 2.0, 0.0, 0.2, 0.6,
                            1.0, 1.1, 0.2, 0.3, -0.2, 0.0
                          },
                          device);

  const StorageView expected_values_k2({2, 2}, std::vector<float>{2.0, 0.6, 1.1, 1.0}, device);
  const StorageView expected_indices_k2({2, 2}, std::vector<int32_t>{2, 5, 1, 0}, device);
  StorageView values_k2(expected_values_k2.dtype(), device);
  StorageView indices_k2(expected_indices_k2.dtype(), device);
  ops::TopK(2)(input, values_k2, indices_k2);
  expect_storage_eq(values_k2, expected_values_k2);
  expect_storage_eq(indices_k2, expected_indices_k2);

  const StorageView expected_values_k3({2, 3}, std::vector<float>{2.0, 0.6, 0.2, 1.1, 1.0, 0.3}, device);
  const StorageView expected_indices_k3({2, 3}, std::vector<int32_t>{2, 5, 4, 1, 0, 3}, device);
  StorageView values_k3(expected_values_k3.dtype(), device);
  StorageView indices_k3(expected_indices_k3.dtype(), device);
  ops::TopK(3)(input, values_k3, indices_k3);
  expect_storage_eq(values_k3, expected_values_k3);
  expect_storage_eq(indices_k3, expected_indices_k3);
}

TEST_P(OpDeviceFPTest, SoftMax) {
  const Device device = GetParam().first;
  const DataType dtype = GetParam().second;
  StorageView x = StorageView({2, 5}, std::vector<float>{
      -0.2, 3.0, 1.2, -1.1, 0.0,
      4.6, 3.3, 0.2, -1.6, 1.0}, device).to(dtype);
  StorageView expected({2, 5}, std::vector<float>{
      0.032035, 0.785904, 0.129909, 0.013025, 0.039128,
      0.760941, 0.207381, 0.009342, 0.001544, 0.020792}, device);
  StorageView y(dtype, device);
  ops::SoftMax()(x, y);
  expect_storage_eq(y.to_float(), expected, 1e-3);
  ops::SoftMax()(x);
  expect_storage_eq(x.to_float(), expected, 1e-3);
}

TEST_P(OpDeviceFPTest, LogSoftMax) {
  const Device device = GetParam().first;
  const DataType dtype = GetParam().second;
  StorageView x = StorageView({2, 10}, std::vector<float>{
      -0.2, 3.0, 1.2, -1.1, 0.0, 0.2, -3.0, -1.2, 1.1, 0.0,
      4.6, 3.3, 0.2, -1.6, 1.0, -4.6, -3.3, -0.2, 1.6, -1.0}, device).to(dtype);
  StorageView expected({2, 10}, std::vector<float>{
      -3.638294, -0.438294, -2.238294, -4.538294, -3.438294, -3.238294, -6.438294, -4.638294, -2.338294, -3.438294,
      -0.319434, -1.619434, -4.719434, -6.519434, -3.919434, -9.519434, -8.219434, -5.119434, -3.319434, -5.919434}, device);
  StorageView y(dtype, device);
  ops::LogSoftMax()(x, y);
  expect_storage_eq(y.to_float(), expected, 1e-2);
  ops::LogSoftMax()(x);
  expect_storage_eq(x.to_float(), expected, 1e-2);
}

TEST_P(OpDeviceFPTest, MaskedSoftMax) {
  const Device device = GetParam().first;
  const DataType dtype = GetParam().second;
  StorageView x({2, 5}, std::vector<float>{
      -0.2, 3.0, 1.2, -1.1, 0.0,
      4.6, 3.3, 0.2, -1.6, 1.0}, device);
  StorageView lengths({2}, std::vector<int32_t>{3, 4}, device);
  StorageView expected({2, 5}, std::vector<float>{
      0.033797, 0.829145, 0.137056,        0, 0,
      0.777098, 0.211783, 0.009540, 0.001577, 0}, device);
  StorageView y(dtype, device);
  ops::SoftMax()(x.to(dtype), lengths, y);
  expect_storage_eq(y.to_float(), expected, 1e-3);
}

TEST_P(OpDeviceFPTest, MaskedSoftMaxTriangular) {
  const Device device = GetParam().first;
  const DataType dtype = GetParam().second;
  StorageView x({2, 2, 3, 3}, std::vector<float>{
      0.08784354, 0.67030656, 0.8866086,
      0.08053982, 0.9826797, 0.7965635,
      0.48865926, 0.8635745, 0.21703207,
      0.0742166, 0.0623771, 0.7590432,
      0.43742728, 0.12613738, 0.53697634,
      0.05396891, 0.04152167, 0.66332567,
      0.6386628, 0.23325896, 0.6977577,
      0.06948507, 0.10246396, 0.6232395,
      0.7822603, 0.3168552, 0.11804962,
      0.1133163, 0.29983068, 0.43074536,
      0.7321733, 0.48709297, 0.35727918,
      0.8421174, 0.9135181, 0.77135813
    }, device);
  StorageView lengths({2}, std::vector<int32_t>{3, 2}, device);
  StorageView mask = layers::MultiHeadAttention::prepare_length_mask(lengths, 2, 3, true);
  StorageView expected({2, 2, 3, 3}, std::vector<float>{
      1, 0, 0,
      0.28861094, 0.71138906, 0,
      0.310848, 0.45224282, 0.23690917,
      1, 0, 0,
      0.57720006, 0.42279992, 0,
      0.26130962, 0.25807717, 0.48061317,
      1, 0, 0,
      0.49175602, 0.508244, 0,
      0.61429566, 0.3857044, 0,
      1, 0, 0,
      0.56096524, 0.43903476, 0,
      0.48215744, 0.5178426, 0
    }, device);
  StorageView y(dtype, device);
  ops::SoftMax()(x.to(dtype), mask, y);
  expect_storage_eq(y.to_float(), expected, 1e-3);
}

TEST_P(OpDeviceFPTest, LayerNorm) {
  const Device device = GetParam().first;
  const DataType dtype = GetParam().second;
  StorageView gamma({5}, std::vector<float>{0.2, 2.1, 1.1, -0.6, 0.7}, device);
  StorageView beta({5}, std::vector<float>{-6.6, -5.7, 0.01, 2.0, 0}, device);
  StorageView x({2, 5}, std::vector<float>{
      -0.2, 3.0, 1.2, -1.1, 0.0,
      4.6, 3.3, 0.2, -1.6, 1.0}, device);
  StorageView expected({2, 5}, std::vector<float>{
      -6.710264, -2.107929, 0.492053, 2.712477, -0.286970,
      -6.319339, -3.988876, -0.637330, 2.841982, -0.158437}, device);
  StorageView y(dtype, device);
  ops::LayerNorm()(beta.to(dtype), gamma.to(dtype), x.to(dtype), y);
  expect_storage_eq(y.to_float(), expected, 1e-3);
}

TEST_P(OpDeviceFPTest, RMSNorm) {
  const Device device = GetParam().first;
  const DataType dtype = GetParam().second;
  StorageView gamma({5}, std::vector<float>{0.2, 2.1, 1.1, -0.6, 0.7}, device);
  StorageView x({2, 5}, std::vector<float>{
      -0.2, 3.0, 1.2, -1.1, 0.0,
      4.6, 3.3, 0.2, -1.6, 1.0}, device);
  StorageView expected({2, 5}, std::vector<float>{
      -0.0262, 4.1202, 0.8633, 0.4316, 0.0000,
      0.3445, 2.5953, 0.0824, 0.3595, 0.2622}, device);
  StorageView y(dtype, device);
  ops::RMSNorm()(gamma.to(dtype), x.to(dtype), y);
  expect_storage_eq(y.to_float(), expected, 1e-3);
}

TEST_P(OpDeviceTest, QuantizeINT8) {
  Device device = GetParam();
  StorageView a({2, 4}, std::vector<float>{-10, -3, 5, 2, 5, 21, -3, 0}, device);
  StorageView scale(DataType::FLOAT32, device);
  StorageView qa(DataType::INT8, device);
  StorageView expected_scale({2}, std::vector<float>{12.7, 6.047619}, device);

  // With rounding before cast.
  {
    StorageView expected_qa(a.shape(), std::vector<int8_t>{-127, -38, 64, 25, 30, 127, -18, 0});
    ops::Quantize(ops::Quantize::ScaleType::GLOBAL, false, true)(a, qa, scale);
    expect_storage_eq(scale, expected_scale);
    expect_storage_eq(qa, expected_qa);
  }

  // Without rounding before cast (legacy behavior).
  {
    StorageView expected_qa(a.shape(), std::vector<int8_t>{-127, -38, 63, 25, 30, 127, -18, 0});
    ops::Quantize(ops::Quantize::ScaleType::GLOBAL, false, false)(a, qa, scale);
    expect_storage_eq(scale, expected_scale);
    expect_storage_eq(qa, expected_qa);
  }
}

TEST_P(OpDeviceTest, QuantizeINT8ZeroRow) {
  Device device = GetParam();
  StorageView a({2, 4}, std::vector<float>{-10, -3, 5, 2, 0, 0, 0, 0}, device);
  StorageView scale(DataType::FLOAT32, device);
  StorageView qa(DataType::INT8, device);
  StorageView expected_scale({2}, std::vector<float>{12.7, 1}, device);

  // With rounding before cast.
  {
    StorageView expected_qa(a.shape(), std::vector<int8_t>{-127, -38, 64, 25, 0, 0, 0, 0});
    ops::Quantize(ops::Quantize::ScaleType::GLOBAL, false, true)(a, qa, scale);
    expect_storage_eq(scale, expected_scale);
    expect_storage_eq(qa, expected_qa);
  }

  // Without rounding before cast (legacy behavior).
  {
    StorageView expected_qa(a.shape(), std::vector<int8_t>{-127, -38, 63, 25, 0, 0, 0, 0});
    ops::Quantize(ops::Quantize::ScaleType::GLOBAL, false, false)(a, qa, scale);
    expect_storage_eq(scale, expected_scale);
    expect_storage_eq(qa, expected_qa);
  }
}

TEST_P(OpDeviceFPTest, Multinomial) {
  const Device device = GetParam().first;
  const DataType dtype = GetParam().second;
  StorageView input({2, 4}, std::vector<float>{0.2, 0.1, 0.6, 0.1, 0.7, 0.2, 0.0, 0.1}, device);
  StorageView output(DataType::INT32, device);
  StorageView counts(input.shape(), int32_t(0));

  constexpr dim_t num_draws = 5000;
  for (dim_t i = 0; i < num_draws; ++i) {
    ops::Multinomial(1)(input.to(dtype), output);
    for (dim_t b = 0; b < output.dim(0); ++b)
      counts.at<int32_t>({b, output.scalar_at<int32_t>({b, 0})}) += 1;
  }

  std::vector<int32_t> counts_vec = counts.to_vector<int32_t>();
  std::vector<float> frequencies(counts_vec.begin(), counts_vec.end());
  for (auto& frequency : frequencies)
    frequency /= num_draws;

  expect_storage_eq(StorageView(input.shape(), frequencies), input, 0.05);
}

TEST_P(OpDeviceFPTest, ReLU) {
  const Device device = GetParam().first;
  const DataType dtype = GetParam().second;
  StorageView input({2, 5}, std::vector<float>{-1, 1, 2, -2, 2, 4, -3, 0, -1, -3}, device);
  StorageView expected({2, 5}, std::vector<float>{0, 1, 2, 0, 2, 4, 0, 0, 0, 0}, device);
  StorageView output(dtype, device);
  ops::ReLU()(input.to(dtype), output);
  expect_storage_eq(output.to_float(), expected);
}

TEST_P(OpDeviceFPTest, GELU) {
  const Device device = GetParam().first;
  const DataType dtype = GetParam().second;
  StorageView input({2}, std::vector<float>{0.2, -1.3}, device);
  StorageView expected({2}, std::vector<float>{0.11585195362567902, -0.1258406937122345}, device);
  StorageView output(dtype, device);
  ops::GELU()(input.to(dtype), output);
  expect_storage_eq(output.to_float(), expected, 1e-4);
}

TEST_P(OpDeviceFPTest, GELUTanh) {
  const Device device = GetParam().first;
  const DataType dtype = GetParam().second;
  StorageView input({2}, std::vector<float>{0.2, -1.3}, device);
  StorageView expected({2}, std::vector<float>{0.11585142463445663, -0.1260710209608078}, device);
  StorageView output(dtype, device);
  const ops::GELU gelu_op(ops::GELU::Approximation::Tanh);
  gelu_op(input.to(dtype), output);
  expect_storage_eq(output.to_float(), expected, 1e-4);
}

TEST_P(OpDeviceFPTest, GELUSigmoid) {
  const Device device = GetParam().first;
  const DataType dtype = GetParam().second;
  StorageView input({2}, std::vector<float>{0.2, -1.3}, device);
  StorageView expected({2}, std::vector<float>{0.11685754358768463, -0.128212109208107}, device);
  StorageView output(dtype, device);
  const ops::GELU gelu_op(ops::GELU::Approximation::Sigmoid);
  gelu_op(input.to(dtype), output);
  expect_storage_eq(output.to_float(), expected, 1e-4);
}

TEST_P(OpDeviceFPTest, Swish) {
  const Device device = GetParam().first;
  const DataType dtype = GetParam().second;
  StorageView input({2}, std::vector<float>{0.2, -1.3}, device);
  StorageView expected({2}, std::vector<float>{0.10996679, -0.27841452}, device);
  StorageView output(dtype, device);
  ops::Swish()(input.to(dtype), output);
  expect_storage_eq(output.to_float(), expected, 1e-4);
}

TEST_P(OpDeviceFPTest, Tanh) {
  const Device device = GetParam().first;
  const DataType dtype = GetParam().second;
  StorageView x({1, 5}, std::vector<float>{-2, -1.5, 0, 1.5, 2}, device);
  StorageView y(dtype, device);
  StorageView expected({1, 5},
                       std::vector<float>{-0.96402758, -0.90514825, 0., 0.90514825, 0.96402758},
                       device);
  ops::Tanh()(x.to(dtype), y);
  expect_storage_eq(y.to_float(), expected, 1e-3);
}

TEST_P(OpDeviceFPTest, Log) {
  const Device device = GetParam().first;
  const DataType dtype = GetParam().second;
  std::vector<float > input_vec({0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4});
  std::vector<float > expected_vec;
  expected_vec.reserve(input_vec.size());
  std::transform(input_vec.begin(), input_vec.end(), std::back_inserter(expected_vec),
          [](const float& i){return std::log(i);});
  StorageView input({2, 4}, input_vec, device);
  StorageView expected({2, 4}, expected_vec, device);
  StorageView output(dtype, device);
  ops::Log()(input.to(dtype), output);
  expect_storage_eq(output.to_float(), expected, 1e-3);
}

TEST_P(OpDeviceFPTest, LogLimits) {
  const Device device = GetParam().first;
  const DataType dtype = GetParam().second;

  StorageView values({2}, std::vector<float>{0.f, -1.f}, device);
  values = values.to(dtype);
  ops::Log()(values, values);
  values = values.to_float();

  EXPECT_EQ(values.scalar_at<float>({0}), -std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::isnan(values.scalar_at<float>({1})));
}

template <typename T, typename Ops, typename Func>
void TestMinMax(Device device, const Ops& ops, const Func& func){
  {
    std::vector<T > input_vec1({0, 1, 1.5, 2, 2.5, 3, 3.5, 4});
    std::vector<T > input_vec2({0, -1, 1.5, -2, 2.5, -3, -3.5, 4});
    std::vector<T > output_vec;
    output_vec.reserve(input_vec1.size());
    std::transform(input_vec1.begin(), input_vec1.end(), input_vec2.begin(),std::back_inserter(output_vec),
            [&func](const T& left, const T& right){return func(left, right);});
    StorageView input1({2, 4}, input_vec1, device);
    StorageView input2({2, 4}, input_vec2, device);
    StorageView expected({2, 4}, output_vec, device);
    StorageView output(device);
    ops(input1, input2, output);
    expect_storage_eq(output, expected);
  }
  {
    std::vector<T > input_vec({0, 1, 1.5, 2, 2.5, 3, 3.5, 4});
    T compare_val = 3;
    std::vector<T > output_vec;
    output_vec.reserve(input_vec.size());
    std::transform(input_vec.begin(), input_vec.end(), std::back_inserter(output_vec),
            [&compare_val, &func](const T& left){return func(left, compare_val);});
    StorageView input({2, 4}, input_vec, device);
    StorageView expected({2, 4}, output_vec, device);
    StorageView output(device);
    ops(input, StorageView(compare_val), output);
    expect_storage_eq(output, expected);
  }
}

TEST_P(OpDeviceTest, Min) {
  Device device = GetParam();
  auto ops = ops::Min();
  TestMinMax<float>(device, ops, [](float left, float right){
    return left > right? right : left;
  });
}

TEST_P(OpDeviceTest, Max) {
  Device device = GetParam();
  auto ops = ops::Max();
  TestMinMax<float>(device, ops, [](float left, float right){
    return left > right? left : right;
  });
}

#ifndef CT2_WITH_CUDNN
#  define GUARD_CONV1D_GPU_TEST GTEST_SKIP() << "Conv1D tests on GPU require cuDNN"
#else
#  define GUARD_CONV1D_GPU_TEST do {} while (0)
#endif

static const StorageView conv_input({2, 2, 3}, std::vector<float>{
    0.5728129f, 0.8784890f, 0.2029965f, 0.3689166f, 0.6570600f, 0.9202735f,
    0.7081605f, 0.3570334f, 0.9339380f, 0.8162224f, 0.0597404f, 0.4628246f});

static const StorageView conv_weight({4, 2, 2}, std::vector<float>{
    0.4969918f, 0.3711241f, 0.1489926f, -0.3010672f,
    -0.2055028f, 0.2540314f, 0.3566069f, -0.1201057f,
    -0.0737700f, -0.0630847f, -0.2370351f, -0.0451550f,
    0.0186623f, 0.3600836f, -0.2889268f, -0.4857445f});

static const StorageView conv_bias({4}, std::vector<float>{
    0.4631361f, -0.1047785f, 0.1047658f, -0.3157263f});

TEST_P(OpDeviceFPTest, Conv1D) {
  const Device device = GetParam().first;
  if (device == Device::CUDA)
    GUARD_CONV1D_GPU_TEST;
  const DataType dtype = GetParam().second;
  const StorageView expected({2, 4, 2}, std::vector<float>{
      0.9309945f, 0.7959076f, 0.0533122f, -0.1099610f,
      -0.1100256f, -0.1701476f, -0.4144599f, -0.8630960f,
      1.0512151f, 0.8567453f, 0.1242856f, 0.0248157f,
      -0.1661695f, -0.0155492f, -0.4387956f, -0.2148425f});
  StorageView output(dtype, device);
  ops::Conv1D()(conv_input.to(device).to(dtype),
                conv_weight.to(device).to(dtype),
                conv_bias.to(device).to(dtype),
                output);
  EXPECT_EQ(output.dtype(), dtype);
  expect_storage_eq(output.to_float(), expected, 1e-3);
}

TEST_P(OpDeviceFPTest, Conv1DNoBias) {
  const Device device = GetParam().first;
  if (device == Device::CUDA)
    GUARD_CONV1D_GPU_TEST;
  const DataType dtype = GetParam().second;
  const StorageView expected({2, 4, 2}, std::vector<float>{
      0.4678584f, 0.3327716f, 0.1580907f, -0.005182412f,
      -0.2147914f, -0.2749133f, -0.09873369f, -0.5473697f,
      0.5880789f, 0.3936091f, 0.2290641f, 0.1295942f,
      -0.2709353f, -0.120315f, -0.1230693f, 0.1008837f});
  StorageView output(dtype, device);
  ops::Conv1D()(conv_input.to(device).to(dtype),
                conv_weight.to(device).to(dtype),
                output);
  EXPECT_EQ(output.dtype(), dtype);
  expect_storage_eq(output.to_float(), expected, 1e-3);
}

TEST_P(OpDeviceFPTest, Conv1DPadding) {
  const Device device = GetParam().first;
  if (device == Device::CUDA)
    GUARD_CONV1D_GPU_TEST;
  const DataType dtype = GetParam().second;
  const StorageView expected({2, 4, 4}, std::vector<float>{
      0.5646521f, 0.9309945f, 0.7959076f, 0.7011377f,
      -0.0035750f, 0.0533122f, -0.1099610f, 0.1816810f,
      0.0519716f, -0.1100256f, -0.1701476f, -0.1283464f,
      -0.2886650f, -0.4144599f, -0.8630960f, -0.5778296f,
      0.4802138f, 1.0512151f, 0.8567453f, 0.9962531f,
      -0.0229165f, 0.1242856f, 0.0248157f, -0.1316590f,
      0.0232352f, -0.1661695f, -0.0155492f, -0.0738365f,
      -0.4572049f, -0.4387956f, -0.2148425f, -0.4320193f});
  StorageView output(dtype, device);
  ops::Conv1D(1, 1)(conv_input.to(device).to(dtype),
                    conv_weight.to(device).to(dtype),
                    conv_bias.to(device).to(dtype),
                    output);
  EXPECT_EQ(output.dtype(), dtype);
  expect_storage_eq(output.to_float(), expected, 1e-3);
}

TEST_P(OpDeviceFPTest, Conv1DStride) {
  const Device device = GetParam().first;
  if (device == Device::CUDA)
    GUARD_CONV1D_GPU_TEST;
  const DataType dtype = GetParam().second;
  const StorageView expected({2, 4, 1}, std::vector<float>{
      0.9309945f, 0.0533122f, -0.1100256f, -0.4144599f,
      1.0512151f, 0.1242856f, -0.1661695f, -0.4387956f});
  StorageView output(dtype, device);
  ops::Conv1D(2)(conv_input.to(device).to(dtype),
                 conv_weight.to(device).to(dtype),
                 conv_bias.to(device).to(dtype),
                 output);
  EXPECT_EQ(output.dtype(), dtype);
  expect_storage_eq(output.to_float(), expected, 1e-3);
}

TEST_P(OpDeviceFPTest, Conv1DPaddingAndStride) {
  const Device device = GetParam().first;
  if (device == Device::CUDA)
    GUARD_CONV1D_GPU_TEST;
  const DataType dtype = GetParam().second;
  const StorageView expected({2, 4, 2}, std::vector<float>{
      0.5646521f, 0.7959076f, -0.0035750f, -0.1099610f,
      0.0519716f, -0.1701476f, -0.2886650f, -0.8630960f,
      0.4802138f, 0.8567453f, -0.0229165f, 0.0248157f,
      0.0232352f, -0.0155492f, -0.4572049f, -0.2148425f});
  StorageView output(dtype, device);
  ops::Conv1D(2, 1)(conv_input.to(device).to(dtype),
                    conv_weight.to(device).to(dtype),
                    conv_bias.to(device).to(dtype),
                    output);
  EXPECT_EQ(output.dtype(), dtype);
  expect_storage_eq(output.to_float(), expected, 1e-3);
}


static std::string fp_test_name(::testing::TestParamInfo<std::pair<Device, DataType>> param_info) {
  return dtype_name(param_info.param.second);
}

INSTANTIATE_TEST_SUITE_P(CPU, OpDeviceTest, ::testing::Values(Device::CPU));
INSTANTIATE_TEST_SUITE_P(CPU, OpDeviceFPTest,
                         ::testing::Values(std::make_pair(Device::CPU, DataType::FLOAT32)),
                         fp_test_name);
#ifdef CT2_WITH_CUDA
INSTANTIATE_TEST_SUITE_P(CUDA, OpDeviceTest, ::testing::Values(Device::CUDA));
INSTANTIATE_TEST_SUITE_P(CUDA, OpDeviceFPTest,
                         ::testing::Values(std::make_pair(Device::CUDA, DataType::FLOAT32),
                                           std::make_pair(Device::CUDA, DataType::FLOAT16)),
                         fp_test_name);
#endif
