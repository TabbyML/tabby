#include <ctranslate2/translator.h>

#include "test_utils.h"

extern std::string g_data_dir;

static std::string path_to_test_name(::testing::TestParamInfo<std::string> param_info) {
  std::string name = param_info.param;
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


// Test that we can load and translate with different versions of the same model.
class ModelVariantTest : public ::testing::TestWithParam<std::string> {
};

TEST_P(ModelVariantTest, Transliteration) {
  Translator translator(g_data_dir + "/models/" + GetParam(), Device::CPU);
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  std::vector<std::string> expected = {"a", "t", "z", "m", "o", "n"};
  auto result = translator.translate(input);
  EXPECT_EQ(result.output(), expected);
}

INSTANTIATE_TEST_CASE_P(
  TranslatorTest,
  ModelVariantTest,
  ::testing::Values(
    "v1/aren-transliteration",
    "v1/aren-transliteration-i16",
    "v2/aren-transliteration",
    "v2/aren-transliteration-i16",
    "v2/aren-transliteration-i8"),
  path_to_test_name);


class SearchVariantTest : public ::testing::TestWithParam<size_t> {
};

TEST_P(SearchVariantTest, SetMaxDecodingLength) {
  Translator translator(g_data_dir + "/models/v2/aren-transliteration", Device::CPU);
  TranslationOptions options;
  options.beam_size = GetParam();
  options.max_decoding_length = 3;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  auto result = translator.translate(input, options);
  EXPECT_EQ(result.output().size(), options.max_decoding_length);
}

TEST_P(SearchVariantTest, SetMinDecodingLength) {
  Translator translator(g_data_dir + "/models/v2/aren-transliteration", Device::CPU);
  TranslationOptions options;
  options.beam_size = GetParam();
  options.min_decoding_length = 8;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  auto result = translator.translate(input, options);
  EXPECT_EQ(result.output().size(), options.min_decoding_length);
}

TEST_P(SearchVariantTest, ReturnAllHypotheses) {
  auto beam_size = GetParam();
  Translator translator(g_data_dir + "/models/v2/aren-transliteration", Device::CPU);
  TranslationOptions options;
  options.beam_size = beam_size;
  options.num_hypotheses = beam_size;
  std::vector<std::string> input = {"آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"};
  auto result = translator.translate(input, options);
  EXPECT_EQ(result.num_hypotheses(), beam_size);
}

INSTANTIATE_TEST_CASE_P(
  TranslatorTest,
  SearchVariantTest,
  ::testing::Values(1, 4),
  beam_to_test_name);
