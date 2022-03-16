#include <ctranslate2/batch_reader.h>
#include <ctranslate2/utils.h>

#include "test_utils.h"

TEST(BatchingTest, RebatchInput) {
  const std::vector<std::vector<std::string>> source = {
    {"a", "b"},
    {"a", "b", "c"},
    {"a"},
    {},
    {"a", "b", "c", "d"},
    {"a", "b", "c", "d", "e"}
  };
  const std::vector<std::vector<std::string>> target = {
    {"1"},
    {"2"},
    {"3"},
    {"4"},
    {"5"},
    {"6"}
  };
  const std::vector<std::vector<size_t>> expected_batches = {
    {5, 4},
    {1, 0},
    {2, 3}
  };

  const auto batches = rebatch_input(load_examples({source, target}), 2, BatchType::Examples);
  ASSERT_EQ(batches.size(), expected_batches.size());

  for (size_t i = 0; i < batches.size(); ++i) {
    const auto& batch = batches[i];
    EXPECT_EQ(batch.get_stream(0), index_vector(source, expected_batches[i]));
    EXPECT_EQ(batch.get_stream(1), index_vector(target, expected_batches[i]));
    EXPECT_EQ(batch.example_index, expected_batches[i]);
  }
}
