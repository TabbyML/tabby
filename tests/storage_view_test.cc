#include "test_utils.h"
#include "ctranslate2/storage_view.h"

TEST(StorageViewTest, Swap) {
  StorageView a({4}, std::vector<float>{1, 2, 3, 4});
  StorageView b({2, 2}, std::vector<float>{2, 3, 4, 5});
  StorageView expected_a(b);
  StorageView expected_b(a);
  const auto* a_data = a.data<float>();
  const auto* b_data = b.data<float>();
  swap(a, b);
  EXPECT_EQ(a.data<float>(), b_data);
  EXPECT_EQ(b.data<float>(), a_data);
  expect_storage_eq(a, expected_a);
  expect_storage_eq(b, expected_b);
}

TEST(StorageViewTest, ZeroDim) {
  StorageView a({2, 0, 2});
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.rank(), 3);
  EXPECT_EQ(a.dim(0), 2);
  EXPECT_EQ(a.dim(1), 0);
  EXPECT_EQ(a.dim(2), 2);
}
