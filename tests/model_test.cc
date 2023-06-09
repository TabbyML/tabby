#include <ctranslate2/models/sequence_to_sequence.h>

#include <ctranslate2/decoding.h>

#include "test_utils.h"

TEST(ModelTest, ContainsModel) {
  ASSERT_TRUE(models::contains_model(default_model_dir()));
}

TEST(ModelTest, UpdateDecoderOutputLayer) {
  auto model = models::Model::load(default_model_dir())->as_sequence_to_sequence();
  auto& decoder = dynamic_cast<models::EncoderDecoderReplica&>(*model).decoder();

  decoder.update_output_layer();
  EXPECT_FALSE(decoder.output_layer_is_updated());
  EXPECT_EQ(decoder.output_size(), 43);

  // Pad to a multiple of 5.
  decoder.update_output_layer(5);
  EXPECT_TRUE(decoder.output_layer_is_updated());
  EXPECT_EQ(decoder.output_size(), 45);

  // Reset output layer.
  decoder.update_output_layer();
  EXPECT_FALSE(decoder.output_layer_is_updated());
  EXPECT_EQ(decoder.output_size(), 43);

  // Restrict to {0, 1, 2, 5} and pad to a multiple of 5.
  decoder.update_output_layer(5, {0, 1, 2, 5});
  EXPECT_TRUE(decoder.output_layer_is_updated());
  EXPECT_EQ(decoder.output_size(), 5);

  EXPECT_TRUE(decoder.is_in_output(0));
  EXPECT_TRUE(decoder.is_in_output(1));
  EXPECT_TRUE(decoder.is_in_output(2));
  EXPECT_FALSE(decoder.is_in_output(4));
  EXPECT_TRUE(decoder.is_in_output(5));

  EXPECT_EQ(decoder.to_original_word_id(0), 0);
  EXPECT_EQ(decoder.to_original_word_id(1), 1);
  EXPECT_EQ(decoder.to_original_word_id(2), 2);
  EXPECT_EQ(decoder.to_original_word_id(3), 5);
  EXPECT_EQ(decoder.to_original_word_id(4), 0);

  // Remove restriction.
  decoder.update_output_layer();
  EXPECT_FALSE(decoder.output_layer_is_updated());
  EXPECT_EQ(decoder.output_size(), 43);
}

TEST(ModelTest, LayerExists) {
  const auto model = models::Model::load(default_model_dir());
  EXPECT_TRUE(model->layer_exists("encoder/layer_0"));
  EXPECT_TRUE(model->layer_exists("encoder/layer_0/"));
  EXPECT_FALSE(model->layer_exists("encoder/layer"));
}

TEST(ModelTest, EncoderDecoderNoLength) {
  auto model = models::Model::load(default_model_dir())->as_sequence_to_sequence();
  auto& encoder_decoder = dynamic_cast<models::EncoderDecoderReplica&>(*model);
  auto& encoder = encoder_decoder.encoder();
  auto& decoder = encoder_decoder.decoder();

  StorageView input_ids({1, 6}, std::vector<int32_t>{31, 10, 19, 13, 5, 7});
  size_t decoder_start_id = 1;
  size_t decoder_end_id = 2;

  std::vector<size_t> output_w_length;
  std::vector<size_t> output_wo_length;

  {
    StorageView lengths({1}, std::vector<int32_t>{6});
    StorageView encoder_output;
    encoder(input_ids, lengths, encoder_output);

    layers::DecoderState state = decoder.initial_state();
    state.emplace("memory", encoder_output);
    state.emplace("memory_lengths", lengths);

    auto results = decode(decoder, state, {{decoder_start_id}}, {decoder_end_id});
    output_w_length = results[0].hypotheses[0];
  }

  {
    StorageView encoder_output;
    encoder(input_ids, encoder_output);

    layers::DecoderState state = decoder.initial_state();
    state.emplace("memory", encoder_output);

    auto results = decode(decoder, state, {{decoder_start_id}}, {decoder_end_id});
    output_wo_length = results[0].hypotheses[0];
  }

  EXPECT_EQ(output_wo_length, output_w_length);
}

TEST(ModelTest, DecoderIterativeSequence) {
  auto model = models::Model::load(default_model_dir())->as_sequence_to_sequence();
  auto& encoder_decoder = dynamic_cast<models::EncoderDecoderReplica&>(*model);
  auto& encoder = encoder_decoder.encoder();
  auto& decoder = encoder_decoder.decoder();

  StorageView source_ids({1, 6}, std::vector<int32_t>{31, 10, 19, 13, 5, 7});
  StorageView target_ids({1, 5}, std::vector<int32_t>{1, 3, 11, 23, 13});

  StorageView encoder_output;
  encoder(source_ids, encoder_output);

  // Forward step by step.
  layers::DecoderState state_by_step = decoder.initial_state();
  state_by_step.emplace("memory", encoder_output);
  StorageView logits_by_step;
  for (dim_t step = 0; step < target_ids.dim(1); ++step) {
    StorageView step_logits;
    StorageView step_input({1}, target_ids.at<int32_t>(step));
    decoder(step, step_input, state_by_step, &step_logits);
    step_logits.expand_dims(1);
    if (step == 0)
      logits_by_step = std::move(step_logits);
    else {
      StorageView logits_concat;
      ops::Concat(1)({&logits_by_step, &step_logits}, logits_concat);
      logits_by_step = std::move(logits_concat);
    }
  }

  // Forward sequence by sequence.
  layers::DecoderState state_sequence = decoder.initial_state();
  state_sequence.emplace("memory", encoder_output);

  StorageView seq1(target_ids.dtype());
  StorageView seq2(target_ids.dtype());
  ops::Split(-1, {3, 2})(target_ids, seq1, seq2);

  StorageView logits1;
  StorageView logits2;
  decoder(0, seq1, state_sequence, &logits1);
  decoder(seq1.dim(-1), seq2, state_sequence, &logits2);

  StorageView logits_sequence;
  ops::Concat(1)({&logits1, &logits2}, logits_sequence);

  expect_storage_eq(logits_sequence, logits_by_step, 1e-5);

  for (const auto& pair : state_by_step) {
    const auto& key = pair.first;
    expect_storage_eq(state_sequence[key], state_by_step[key], 1e-5);
  }
}
