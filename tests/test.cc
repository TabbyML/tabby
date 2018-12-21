#include <gtest/gtest.h>

#include "test_utils.h"

std::string g_data_dir;

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  if (argc < 2)
    throw std::invalid_argument("missing data directory");
  g_data_dir = argv[1];
  return RUN_ALL_TESTS();
}
