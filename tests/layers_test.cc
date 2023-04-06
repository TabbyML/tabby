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

TEST(LayerTest, Alibi) {
  const StorageView lengths({3}, std::vector<int32_t>{3, 5, 2});
  const StorageView alibi = layers::build_alibi(3, 4, 2, 5, &lengths);
  const StorageView expected({3, 4, 2, 5}, std::vector<float>{
      0.0000, 0.2500, 0.5000, 0.0000, 0.0000,
      0.0000, 0.2500, 0.5000, 0.0000, 0.0000,
      0.0000, 0.0625, 0.1250, 0.0000, 0.0000,
      0.0000, 0.0625, 0.1250, 0.0000, 0.0000,
      0.0000, 0.0156, 0.0312, 0.0000, 0.0000,
      0.0000, 0.0156, 0.0312, 0.0000, 0.0000,
      0.0000, 0.0039, 0.0078, 0.0000, 0.0000,
      0.0000, 0.0039, 0.0078, 0.0000, 0.0000,
      0.0000, 0.2500, 0.5000, 0.7500, 1.0000,
      0.0000, 0.2500, 0.5000, 0.7500, 1.0000,
      0.0000, 0.0625, 0.1250, 0.1875, 0.2500,
      0.0000, 0.0625, 0.1250, 0.1875, 0.2500,
      0.0000, 0.0156, 0.0312, 0.0469, 0.0625,
      0.0000, 0.0156, 0.0312, 0.0469, 0.0625,
      0.0000, 0.0039, 0.0078, 0.0117, 0.0156,
      0.0000, 0.0039, 0.0078, 0.0117, 0.0156,
      0.0000, 0.2500, 0.0000, 0.0000, 0.0000,
      0.0000, 0.2500, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0625, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0625, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0156, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0156, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0039, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0039, 0.0000, 0.0000, 0.0000});

  expect_storage_eq(alibi, expected, 1e-4);
}

TEST(LayerTest, RotaryEmbedding) {
  StorageView x({2, 4, 1, 4}, std::vector<float>{
      0.8822692632675171, 0.9150039553642273, 0.38286375999450684, 0.9593056440353394,
      0.3904482126235962, 0.600895345211029, 0.2565724849700928, 0.7936413288116455,
      0.9407714605331421, 0.13318592309951782, 0.9345980882644653, 0.5935796499252319,
      0.8694044351577759, 0.5677152872085571, 0.7410940527915955, 0.42940449714660645,
      0.8854429125785828, 0.5739044547080994, 0.2665800452232361, 0.6274491548538208,
      0.26963168382644653, 0.4413635730743408, 0.2969208359718323, 0.831685483455658,
      0.10531491041183472, 0.26949483156204224, 0.3588126301765442, 0.19936376810073853,
      0.5471915602684021, 0.006160438060760498, 0.951554536819458, 0.07526588439941406
    });

  StorageView expected({2, 4, 1, 4}, std::vector<float>{
      -1.1991642713546753, 0.421469122171402, 0.3636023700237274, 0.966770589351654,
      -0.708876371383667, 0.10497283935546875, 0.24064940214157104, 0.7986137270927429,
      -0.5126047134399414, 0.8000161051750183, 0.9225403666496277, 0.6121516823768616,
      -0.8780219554901123, 0.5542942881584167, 0.7323583364486694, 0.4441395401954651,
      -0.8903241157531738, 0.5663024187088013, 0.25397858023643494, 0.6326549053192139,
      -0.5135371088981628, 0.0615033358335495, 0.2802288830280304, 0.8374571800231934,
      -0.2888774275779724, -0.01638685166835785, 0.35475385189056396, 0.20649968087673187,
      -0.23331370949745178, 0.49499621987342834, 0.949859082698822, 0.0942806527018547
    });

  layers::RotaryEmbeddings rotary_embeddings;
  rotary_embeddings.apply(x, 2);
  expect_storage_eq(x, expected, 1e-6);
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
