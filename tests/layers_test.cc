#include "test_utils.h"
#include "ctranslate2/layers/layers.h"
#include "ctranslate2/padder.h"

TEST(LayerTest, MakeRelativePositions1D) {
  const StorageView positions = layers::make_relative_positions(4, 2, true);
  const StorageView expected({1, 4}, std::vector<int32_t>{0, 0, 1, 2});
  expect_storage_eq(positions, expected);
}

TEST(LayerTest, MakeRelativePositions2D) {
  const StorageView positions = layers::make_relative_positions(4, 2);
  const StorageView expected({4, 4}, std::vector<int32_t>{
      2, 3, 4, 4,
      1, 2, 3, 4,
      0, 1, 2, 3,
      0, 0, 1, 2});
  expect_storage_eq(positions, expected);
}

TEST(LayerTest, Padder) {
  const StorageView lengths({3}, std::vector<int32_t>{2, 3, 1});
  const Padder padder(lengths, /*max_time=*/4);

  StorageView x({3, 4}, std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  const StorageView wo_padding({6}, std::vector<int32_t>{0, 1, 4, 5, 6, 8});
  const StorageView w_padding({3, 4}, std::vector<int32_t>{0, 1, 1, 1, 4, 5, 6, 6, 8, 8, 8, 8});

  padder.remove_padding(x);
  ASSERT_EQ(x.rank(), 1);
  expect_storage_eq(x, wo_padding);
  padder.add_padding(x);
  ASSERT_EQ(x.rank(), 2);
  expect_storage_eq(x, w_padding);
}

TEST(LayerTest, PadderToMultiple) {
  const StorageView lengths({3}, std::vector<int32_t>{2, 3, 1});
  const Padder padder(lengths, /*max_time=*/4, /*pad_batch_to_multiple=*/8);

  StorageView x({3, 4}, std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  const StorageView wo_padding({8}, std::vector<int32_t>{0, 1, 4, 5, 6, 8, 8, 8});
  const StorageView w_padding({3, 4}, std::vector<int32_t>{0, 1, 1, 1, 4, 5, 6, 6, 8, 8, 8, 8});

  padder.remove_padding(x);
  expect_storage_eq(x, wo_padding);
  padder.add_padding(x);
  expect_storage_eq(x, w_padding);
}

TEST(LayerTest, PadderIgnore) {
  const StorageView lengths({3}, std::vector<int32_t>{4, 4, 4});
  const Padder padder(lengths);

  StorageView x({3, 4}, std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  const StorageView original(x);

  padder.remove_padding(x);
  expect_storage_eq(x, original);
  padder.add_padding(x);
  expect_storage_eq(x, original);
}

TEST(LayerTest, PositionEncoderNoSharedState) {
  // Test case for issue: http://forum.opennmt.net/t/ctranslate2-c-api-returns-strange-results-when-initializing-2-models/3208
  layers::SinusoidalPositionEncoder position_encoder_1(4);
  layers::SinusoidalPositionEncoder position_encoder_2(6);

  {
    StorageView input(
      {1, 1, 4}, std::vector<float>{0.1, -2.3, 0.5, 1.2});
    StorageView expected(
      {1, 1, 4}, std::vector<float>{0.941471, -2.2999, 1.0403, 2.2});
    position_encoder_1(input);
    expect_storage_eq(input, expected, 1e-5);
  }

  {
    StorageView input(
      {1, 1, 6}, std::vector<float>{-0.2, -1.3, 0.1, -0.6, 2.0, 1.1});
    StorageView expected(
      {1, 1, 6}, std::vector<float>{0.641471, -1.29, 0.1001, -0.0596977, 2.99995, 2.1});
    position_encoder_2(input);
    expect_storage_eq(input, expected, 1e-5);
  }
}
