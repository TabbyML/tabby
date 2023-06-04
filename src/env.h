#pragma once

#include <string>

namespace ctranslate2 {

  std::string read_string_from_env(const char* var, const std::string& default_value = "");
  bool read_bool_from_env(const char* var, const bool default_value = false);
  int read_int_from_env(const char* var, const int default_value = 0);

}
