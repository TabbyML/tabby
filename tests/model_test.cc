#include <ctranslate2/models/model.h>

#include "test_utils.h"

extern std::string g_data_dir;

TEST(ModelTest, ContainsModel) {
  ASSERT_TRUE(models::contains_model(g_data_dir + "/models/v2/aren-transliteration"));
}
