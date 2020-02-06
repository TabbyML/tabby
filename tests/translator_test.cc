#include <ctranslate2/translator.h>

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


// Test that we can load and translate with different versions of the same model.
class ModelVariantTest : public ::testing::TestWithParam<std::pair<std::string, DataType>> {
};

TEST_P(ModelVariantTest, Transliteration) {
  auto params = GetParam();
  const std::string& model_path = params.first;
  DataType expected_dtype = params.second;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  std::vector<std::string> expected = {"a", "t", "z", "m", "o", "n"};

  std::vector<std::pair<ComputeType, DataType>> type_params;
  type_params.emplace_back(std::pair<ComputeType, DataType>(ComputeType::DEFAULT, expected_dtype));
  type_params.emplace_back(std::pair<ComputeType, DataType>(ComputeType::FLOAT, DataType::DT_FLOAT));
  type_params.emplace_back(std::pair<ComputeType, DataType>(ComputeType::INT16, DataType::DT_INT16));
  type_params.emplace_back(std::pair<ComputeType, DataType>(ComputeType::INT8, DataType::DT_INT8));

  for (const auto& t : type_params) {
    // compute type: none
    auto model = models::Model::load(g_data_dir + "/models/" + model_path, Device::CPU, 0, t.first);
    check_weights_dtype(model->get_variables(), t.second);
    Translator translator(model);
    auto result = translator.translate(input);
    EXPECT_EQ(result.output(), expected);
  }
}

INSTANTIATE_TEST_CASE_P(
  TranslatorTest,
  ModelVariantTest,
  ::testing::Values(
    std::make_pair("v1/aren-transliteration", DataType::DT_FLOAT),
    std::make_pair("v1/aren-transliteration-i16", DataType::DT_INT16),
    std::make_pair("v2/aren-transliteration", DataType::DT_FLOAT),
    std::make_pair("v2/aren-transliteration-i16", DataType::DT_INT16),
    std::make_pair("v2/aren-transliteration-i8", DataType::DT_INT8)
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
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  auto result = translator.translate(input, options);
  ASSERT_TRUE(result.has_attention());
  const auto& attention = result.attention();
  EXPECT_EQ(attention.size(), beam_size);
  EXPECT_EQ(attention[0].size(), 6);
  EXPECT_EQ(attention[0][0].size(), 6);
}

TEST_P(SearchVariantTest, TranslateWithPrefix) {
  Translator translator = default_translator();
  TranslationOptions options;
  options.beam_size = GetParam();
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  std::vector<std::string> prefix = {"a", "t", "z"};
  std::vector<std::string> expected = {"a", "t", "z", "m", "o", "n"};
  auto result = translator.translate_with_prefix(input, prefix, options);
  EXPECT_EQ(result.output(), expected);
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
