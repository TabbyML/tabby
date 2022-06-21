#include <ctranslate2/models/sequence_to_sequence.h>

#include "test_utils.h"

TEST(ModelTest, ContainsModel) {
  ASSERT_TRUE(models::contains_model(default_model_dir()));
}

TEST(ModelTest, UpdateDecoderOutputLayer) {
  auto model = models::Model::load(default_model_dir())->as_sequence_to_sequence();
  auto& decoder = dynamic_cast<models::EncoderDecoderReplica&>(*model).decoder();

  EXPECT_EQ(decoder.update_output_layer(), nullptr);
  EXPECT_EQ(decoder.output_size(), 43);

  // Pad to a multiple of 5.
  EXPECT_NE(decoder.update_output_layer(5), nullptr);
  EXPECT_EQ(decoder.output_size(), 45);

  // Exclude index 1 and pad to a multiple of 2.
  EXPECT_NE(decoder.update_output_layer(2, {}, {1}), nullptr);
  EXPECT_EQ(decoder.output_size(), 42);

  // Exclude index 1 and pad to a multiple of 43.
  EXPECT_NE(decoder.update_output_layer(43, {}, {1}), nullptr);
  EXPECT_EQ(decoder.output_size(), 43);

  // Reset output layer.
  EXPECT_EQ(decoder.update_output_layer(), nullptr);
  EXPECT_EQ(decoder.output_size(), 43);

  // Restrict to {0, 1, 2, 5} - {1} and pad to a multiple of 5.
  EXPECT_EQ(*decoder.update_output_layer(5, {0, 1, 2, 5}, {1}),
            (std::vector<size_t>{0, 2, 5, 0, 0}));
  EXPECT_EQ(decoder.output_size(), 5);

  // Remove restriction.
  EXPECT_EQ(decoder.update_output_layer(), nullptr);
  EXPECT_EQ(decoder.output_size(), 43);
}

TEST(ModelTest, LayerExists) {
  const auto model = models::Model::load(default_model_dir());
  EXPECT_TRUE(model->layer_exists("encoder/layer_0"));
  EXPECT_TRUE(model->layer_exists("encoder/layer_0/"));
  EXPECT_FALSE(model->layer_exists("encoder/layer"));
}
