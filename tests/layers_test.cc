#include "test_utils.h"
#include "ctranslate2/layers/layers.h"

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
