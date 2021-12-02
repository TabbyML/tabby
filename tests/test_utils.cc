#include "test_utils.h"

extern std::string g_data_dir;

const std::string& get_data_dir() {
  return g_data_dir;
}

std::string default_model_dir() {
  return g_data_dir + "/models/v2/aren-transliteration";
}
