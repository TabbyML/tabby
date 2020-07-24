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
