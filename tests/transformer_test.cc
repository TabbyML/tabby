#include <ctranslate2/models/transformer.h>

#include "test_utils.h"

extern std::string g_data_dir;

TEST(TransformerTest, PositionEncoderNoSharedState) {
  // Test case for issue: http://forum.opennmt.net/t/ctranslate2-c-api-returns-strange-results-when-initializing-2-models/3208
  models::PositionEncoder position_encoder_1;
  models::PositionEncoder position_encoder_2;

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
