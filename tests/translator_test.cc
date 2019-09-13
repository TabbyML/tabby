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

static bool endswith(const std::string& str, const std::string& part) {
  return str.size() >= part.size() && str.substr(str.size() - part.size()) == part;
}

static void check_weights_dtype(const std::unordered_map<std::string, StorageView>& variables,
                                DataType expected_dtype) {
  for (const auto& variable : variables) {
    const auto& name = variable.first;
    const auto& value = variable.second;
    if (endswith(name, "weight")) {
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

  // compute type: none
  auto model_0 = models::ModelFactory::load(g_data_dir + "/models/" + model_path, Device::CPU, "none");
  check_weights_dtype(model_0->get_variables(), expected_dtype);
  Translator translator_0(model_0);
  auto result_0 = translator_0.translate(input);
  EXPECT_EQ(result_0.output(), expected);

  // compute type: int8
  auto model_1 = models::ModelFactory::load(g_data_dir + "/models/" + model_path, Device::CPU, "int8");
  check_weights_dtype(model_1->get_variables(), DataType::DT_INT8);
  Translator translator_1(model_1);
  auto result_1 = translator_1.translate(input);
  EXPECT_EQ(result_1.output(), expected);

  // compute type: int16
  auto model_2 = models::ModelFactory::load(g_data_dir + "/models/" + model_path, Device::CPU, "int16");
  check_weights_dtype(model_2->get_variables(), DataType::DT_INT16);
  Translator translator_2(model_2);
  auto result_2 = translator_2.translate(input);
  EXPECT_EQ(result_2.output(), expected);

  // compute type: float
  auto model_3 = models::ModelFactory::load(g_data_dir + "/models/" + model_path, Device::CPU, "float");
  check_weights_dtype(model_3->get_variables(), DataType::DT_FLOAT);
  Translator translator_3(model_3);
  auto result_3 = translator_3.translate(input);
  EXPECT_EQ(result_3.output(), expected);

}

INSTANTIATE_TEST_CASE_P(
  TranslatorTest,
  ModelVariantTest,
  ::testing::Values(
    std::make_pair("v1/aren-transliteration", DataType::DT_FLOAT),
    std::make_pair("v1/aren-transliteration-i16", DataType::DT_INT16),
    std::make_pair("v2/aren-transliteration", DataType::DT_FLOAT),
    std::make_pair("v2/aren-transliteration-i16", DataType::DT_INT16),
#ifdef WITH_MKLDNN
    std::make_pair("v2/aren-transliteration-i8", DataType::DT_INT8)
#else
    std::make_pair("v2/aren-transliteration-i8", DataType::DT_INT16)
#endif
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

INSTANTIATE_TEST_CASE_P(
  TranslatorTest,
  SearchVariantTest,
  ::testing::Values(1, 4),
  beam_to_test_name);
