#include <ctranslate2/translator.h>

#include "test_utils.h"

extern std::string g_data_dir;

static std::string path_to_test_name(::testing::TestParamInfo<std::string> param_info) {
  std::string name = param_info.param;
  std::replace(name.begin(), name.end(), '/', '_');
  std::replace(name.begin(), name.end(), '-', '_');
  return name;
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
