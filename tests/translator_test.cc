#include <ctranslate2/translator.h>

#include <algorithm>

#include "test_utils.h"

extern std::string g_data_dir;

static std::string
path_to_test_name(::testing::TestParamInfo<std::pair<std::string, DataType>> param_info) {
  std::string name = param_info.param.first;
  std::replace(name.begin(), name.end(), '/', '_');
  std::replace(name.begin(), name.end(), '-', '_');
  return name;
}

static std::string beam_to_test_name(::testing::TestParamInfo<size_t> param_info) {
  if (param_info.param == 1)
    return "GreedySearch";
  else
    return "BeamSearch";
}

static void check_weights_dtype(const std::unordered_map<std::string, StorageView>& variables,
                                DataType expected_dtype) {
  for (const auto& variable : variables) {
    const auto& name = variable.first;
    const auto& value = variable.second;
    if (ends_with(name, "weight")) {
      EXPECT_EQ(value.dtype(), expected_dtype) << "Expected type " << dtype_name(expected_dtype)
                                               << " for weight " << name << ", got "
                                               << dtype_name(value.dtype()) << " instead";
    }
  }
}

static DataType dtype_with_fallback(DataType dtype, Device device) {
  const bool support_int8 = mayiuse_int8(device);
  const bool support_int16 = mayiuse_int16(device);
  if (dtype == DataType::INT16 && !support_int16)
    return support_int8 ? DataType::INT8 : DataType::FLOAT;
  if (dtype == DataType::INT8 && !support_int8)
    return support_int16 ? DataType::INT16 : DataType::FLOAT;
  return dtype;
}


// Test that we can load and translate with different versions of the same model.
class ModelVariantTest : public ::testing::TestWithParam<std::pair<std::string, DataType>> {
};

TEST_P(ModelVariantTest, Transliteration) {
  auto params = GetParam();
  const std::string model_path = g_data_dir + "/models/" + params.first;
  const DataType model_dtype = params.second;
  const Device device = Device::CPU;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  std::vector<std::string> expected = {"a", "t", "z", "m", "o", "n"};

  std::vector<std::pair<ComputeType, DataType>> type_params;
  type_params.emplace_back(ComputeType::DEFAULT, dtype_with_fallback(model_dtype, device));
  type_params.emplace_back(ComputeType::FLOAT, DataType::FLOAT);
  if (mayiuse_int16(device))
    type_params.emplace_back(ComputeType::INT16, DataType::INT16);
  if (mayiuse_int8(device))
    type_params.emplace_back(ComputeType::INT8, DataType::INT8);

  for (const auto& types : type_params) {
    const ComputeType compute_type = types.first;
    const DataType expected_type = types.second;
    const auto model = models::Model::load(model_path, device, 0, compute_type);
    check_weights_dtype(model->get_variables(), expected_type);
    Translator translator(model);
    auto result = translator.translate(input);
    EXPECT_EQ(result.output(), expected);
  }
}

INSTANTIATE_TEST_CASE_P(
  TranslatorTest,
  ModelVariantTest,
  ::testing::Values(
    std::make_pair("v1/aren-transliteration", DataType::FLOAT),
    std::make_pair("v1/aren-transliteration-i16", DataType::INT16),
    std::make_pair("v2/aren-transliteration", DataType::FLOAT),
    std::make_pair("v2/aren-transliteration-i16", DataType::INT16),
    std::make_pair("v2/aren-transliteration-i8", DataType::INT8)
    ),
  path_to_test_name);


class SearchVariantTest : public ::testing::TestWithParam<size_t> {
};

static Translator default_translator() {
  return Translator(g_data_dir + "/models/v2/aren-transliteration", Device::CPU);
}

TEST_P(SearchVariantTest, SetMaxDecodingLength) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = GetParam();
  options.max_decoding_length = 3;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  auto result = translator.translate(input, options);
  EXPECT_EQ(result.output().size(), options.max_decoding_length);
}

TEST_P(SearchVariantTest, SetMinDecodingLength) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = GetParam();
  options.min_decoding_length = 8;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  auto result = translator.translate(input, options);
  EXPECT_EQ(result.output().size(), options.min_decoding_length);
}

TEST_P(SearchVariantTest, ReturnAllHypotheses) {
  auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.num_hypotheses = beam_size;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  auto result = translator.translate(input, options);
  EXPECT_EQ(result.num_hypotheses(), beam_size);
}

TEST_P(SearchVariantTest, ReturnAttention) {
  auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.num_hypotheses = beam_size;
  options.return_attention = true;
  const std::vector<std::vector<std::string>> inputs = {
    {"آ", "ز", "ا"},
    {"آ", "ت", "ز", "م", "و", "ن"}
  };
  const std::vector<std::pair<size_t, size_t>> expected_shapes = {
    {4, 3},
    {6, 6}
  };  // (target_length, source_length)
  const auto results = translator.translate_batch(inputs, options);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const TranslationResult& result = results[i];
    const auto& expected_shape = expected_shapes[i];
    ASSERT_TRUE(result.has_attention());
    const auto& attention = result.attention();
    EXPECT_EQ(attention.size(), beam_size);
    EXPECT_EQ(attention[0].size(), expected_shape.first);
    for (const auto& vector : attention[0]) {
      EXPECT_EQ(vector.size(), expected_shape.second);
    }
  }
}

TEST_P(SearchVariantTest, TranslateWithPrefix) {
  const auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.num_hypotheses = beam_size;
  options.return_attention = true;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  std::vector<std::string> prefix = {"a", "t", "s"};
  std::vector<std::string> expected = {"a", "t", "s", "u", "m", "o", "n"};
  auto result = translator.translate_with_prefix(input, prefix, options);
  EXPECT_EQ(result.num_hypotheses(), beam_size);
  EXPECT_EQ(result.output(), expected);
  ASSERT_TRUE(result.has_attention());
  const auto& attention = result.attention();
  EXPECT_EQ(attention.size(), options.beam_size);
  EXPECT_EQ(attention[0].size(), 7);
  EXPECT_EQ(attention[0][0].size(), 6);
}

TEST_P(SearchVariantTest, TranslateBatch) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = GetParam();
  std::vector<std::vector<std::string>> inputs = {
    {"آ", "ز", "ا"},
    {"آ", "ت", "ز", "م", "و", "ن"}};
  std::vector<std::vector<std::string>> expected = {
    {"a", "z", "z", "a"},
    {"a", "t", "z", "m", "o", "n"}};
  auto result = translator.translate_batch(inputs, options);
  EXPECT_EQ(result[0].output(), expected[0]);
  EXPECT_EQ(result[1].output(), expected[1]);
}

TEST_P(SearchVariantTest, ReplaceUnknowns) {
  const auto beam_size = GetParam();
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = beam_size;
  options.num_hypotheses = beam_size;
  options.replace_unknowns = true;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  std::vector<std::string> prefix = {"<unk>", "t"};
  std::vector<std::string> expected = {"ت", "t", "z", "m", "o", "n" };
  auto result = translator.translate_with_prefix(input, prefix, options);
  EXPECT_EQ(result.output(), expected);
}


INSTANTIATE_TEST_CASE_P(
  TranslatorTest,
  SearchVariantTest,
  ::testing::Values(1, 4),
  beam_to_test_name);

TEST(TranslatorTest, TranslateBatchWithMaxBatchSize) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.max_batch_size = 2;
  std::vector<std::vector<std::string>> inputs = {
    {"آ" ,"ر" ,"ب" ,"ا" ,"ك" ,"ه"},
    {"آ" ,"ز" ,"ا"},
    {"آ" ,"ت" ,"ش" ,"ي" ,"س" ,"و" ,"ن"},
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ر" ,"ث" ,"ر"}};
  auto results = translator.translate_batch(inputs, options);
  ASSERT_EQ(results.size(), 5);
  // Order should be preserved.
  EXPECT_EQ(results[0].output(), (std::vector<std::string>{"a", "r", "b", "a", "k", "e"}));
  EXPECT_EQ(results[1].output(), (std::vector<std::string>{"a", "z", "z", "a"}));
  EXPECT_EQ(results[2].output(), (std::vector<std::string>{"a", "c", "h", "i", "s", "o", "n"}));
  EXPECT_EQ(results[3].output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(results[4].output(), (std::vector<std::string>{"a", "r", "t", "h", "e", "r"}));
}

TEST(TranslatorTest, TranslateEmptyBatch) {
  Translator translator = default_translator();
  std::vector<std::vector<std::string>> inputs;
  auto results = translator.translate_batch(inputs);
  EXPECT_TRUE(results.empty());
}

static void check_empty_result(const TranslationResult& result,
                               size_t num_hypotheses = 1,
                               bool with_attention = false) {
  EXPECT_TRUE(result.output().empty());
  EXPECT_EQ(result.score(), static_cast<float>(0));
  EXPECT_EQ(result.num_hypotheses(), num_hypotheses);
  EXPECT_EQ(result.hypotheses().size(), num_hypotheses);
  EXPECT_EQ(result.scores().size(), num_hypotheses);
  EXPECT_EQ(result.has_attention(), with_attention);
  if (with_attention) {
    const auto& attention = result.attention();
    EXPECT_EQ(attention.size(), num_hypotheses);
    EXPECT_TRUE(attention[0].empty());
  }
}

TEST(TranslatorTest, TranslateBatchWithEmptySource) {
  Translator translator = default_translator();
  std::vector<std::vector<std::string>> inputs = {
    {}, {"آ", "ز", "ا"}, {}, {"آ", "ت", "ز", "م", "و", "ن"}, {}};
  auto results = translator.translate_batch(inputs);
  EXPECT_EQ(results.size(), 5);
  check_empty_result(results[0]);
  EXPECT_EQ(results[1].output(), (std::vector<std::string>{"a", "z", "z", "a"}));
  check_empty_result(results[2]);
  EXPECT_EQ(results[3].output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  check_empty_result(results[4]);
}

TEST(TranslatorTest, TranslateBatchWithOnlyEmptySource) {
  Translator translator = default_translator();
  std::vector<std::vector<std::string>> inputs{{}, {}};
  auto results = translator.translate_batch(inputs);
  EXPECT_EQ(results.size(), 2);
  check_empty_result(results[0]);
  check_empty_result(results[1]);
}

TEST(TranslatorTest, TranslateBatchWithPrefixAndEmpty) {
  Translator translator = default_translator();
  const TranslationOptions options;
  const std::vector<std::vector<std::string>> input = {
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {},
    {"آ" ,"ز" ,"ا"}};
  const std::vector<std::vector<std::string>> prefix = {
    {"a", "t", "s"},
    {},
    {"a", "t", "z", "o"},
    {},
    {}};
  const auto result = translator.translate_batch_with_prefix(input, prefix, options);
  EXPECT_EQ(result[0].output(), (std::vector<std::string>{"a", "t", "s", "u", "m", "o", "n"}));
  EXPECT_EQ(result[1].output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(result[2].output(), (std::vector<std::string>{"a", "t", "z", "o", "m", "o", "n"}));
  EXPECT_TRUE(result[3].output().empty());
  EXPECT_EQ(result[4].output(), (std::vector<std::string>{"a", "z", "z", "a"}));
}

TEST(TranslatorTest, TranslatePrefixWithLargeBeam) {
  // Related to issue https://github.com/OpenNMT/CTranslate2/issues/277
  // This is an example where </s> appears in the topk of the first unconstrained decoding
  // step and produces an incorrect hypothesis that dominates others.
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = 5;
  const std::vector<std::string> input = {"أ" ,"و" ,"ل" ,"ي" ,"س" ,"س"};
  const std::vector<std::string> prefix = {"u", "l", "i", "s", "e"};
  const auto result = translator.translate_with_prefix(input, prefix, options);
  EXPECT_EQ(result.output(), (std::vector<std::string>{"u", "l", "i", "s", "e", "s"}));
}

TEST(TranslatorTest, AlternativesFromPrefix) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 10;
  options.return_alternatives = true;
  options.return_attention = true;
  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  const std::vector<std::string> prefix = {"a", "t"};
  const TranslationResult result = translator.translate_with_prefix(input, prefix, options);
  ASSERT_EQ(result.num_hypotheses(), options.num_hypotheses);
  EXPECT_EQ(result.hypotheses()[0], (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(result.hypotheses()[1], (std::vector<std::string>{"a", "t", "s", "u", "m", "o", "n"}));

  // Tokens at the first unconstrained decoding position should be unique.
  std::vector<std::string> tokens_at_position;
  tokens_at_position.reserve(options.num_hypotheses);
  for (const std::vector<std::string>& hypothesis : result.hypotheses())
    tokens_at_position.emplace_back(hypothesis[prefix.size()]);
  EXPECT_EQ(std::unique(tokens_at_position.begin(), tokens_at_position.end()),
            tokens_at_position.end());

  EXPECT_TRUE(result.has_attention());
  EXPECT_EQ(result.attention()[0].size(), 6);
}

TEST(TranslatorTest, AlternativesFromScratch) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 10;
  options.return_alternatives = true;
  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  const TranslationResult result = translator.translate(input, options);
  ASSERT_EQ(result.num_hypotheses(), options.num_hypotheses);
  EXPECT_EQ(result.hypotheses()[0], (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
}

TEST(TranslatorTest, AlternativesFromScratchBatch) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 10;
  options.return_alternatives = true;
  const std::vector<std::vector<std::string>> inputs = {
    {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"},
    {"آ", "ز", "ا"}
  };
  const std::vector<TranslationResult> results = translator.translate_batch(inputs, options);
  ASSERT_EQ(results.size(), inputs.size());
  ASSERT_EQ(results[0].num_hypotheses(), options.num_hypotheses);
  EXPECT_EQ(results[0].hypotheses()[0], (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
  EXPECT_EQ(results[0].hypotheses()[1], (std::vector<std::string>{"e", "t", "z", "m", "o", "n"}));
  ASSERT_EQ(results[1].num_hypotheses(), options.num_hypotheses);
  EXPECT_EQ(results[1].hypotheses()[0], (std::vector<std::string>{"a", "z", "z", "a"}));
  EXPECT_EQ(results[1].hypotheses()[1], (std::vector<std::string>{"e", "z", "a"}));
}

TEST(TranslatorTest, AlternativesFromFullTarget) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 4;
  options.return_alternatives = true;
  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  const std::vector<std::string> target = {"a", "t", "z", "m", "o", "n"};
  const TranslationResult result = translator.translate_with_prefix(input, target, options);
  EXPECT_EQ(result.hypotheses()[0], (std::vector<std::string>{"a", "t", "z", "m", "o", "n", "e"}));
}

TEST(TranslatorTest, DetachModel) {
  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  Translator translator = default_translator();
  translator.detach_model();
  EXPECT_THROW(translator.translate(input), std::runtime_error);
  Translator clone(translator);
  EXPECT_THROW(clone.translate(input), std::runtime_error);
  translator.set_model(models::Model::load(g_data_dir + "/models/v2/aren-transliteration",
                                           Device::CPU));
  translator.translate(input);
}

TEST(TranslatorTest, InvalidNumHypotheses) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.num_hypotheses = 0;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  EXPECT_THROW(translator.translate(input, options), std::invalid_argument);
}

TEST(TranslatorTest, IgnoreScore) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = 1;
  options.return_scores = false;
  const std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  const TranslationResult result = translator.translate(input, options);
  EXPECT_FALSE(result.has_scores());
  EXPECT_EQ(result.output(), (std::vector<std::string>{"a", "t", "z", "m", "o", "n"}));
}
