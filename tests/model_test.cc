#include <ctranslate2/models/model.h>

#include "test_utils.h"

TEST(ModelTest, ContainsModel) {
  ASSERT_TRUE(models::contains_model(default_model_dir()));
}
