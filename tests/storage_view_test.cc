#include "test_utils.h"
#include "opennmt/storage_view.h"

TEST(StorageViewTest, Swap) {
  StorageView a({4}, std::vector<float>{1, 2, 3, 4});
  StorageView b({2, 2}, std::vector<float>{2, 3, 4, 5});
  StorageView expected_a(b);
  StorageView expected_b(a);
  const auto* a_data = a.data<float>();
  const auto* b_data = b.data<float>();
  std::swap(a, b);
  EXPECT_EQ(a.data<float>(), b_data);
  EXPECT_EQ(b.data<float>(), a_data);
  expect_storage_eq(a, expected_a);
  expect_storage_eq(b, expected_b);
}
