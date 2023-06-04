#include "env.h"

#include <cstdlib>

#include "ctranslate2/utils.h"

namespace ctranslate2 {

  std::string read_string_from_env(const char* var, const std::string& default_value) {
    const char* value = std::getenv(var);
    if (!value)
      return default_value;
    return value;
  }

  bool read_bool_from_env(const char* var, const bool default_value) {
    return string_to_bool(read_string_from_env(var, default_value ? "1" : "0"));
  }

  int read_int_from_env(const char* var, const int default_value) {
    const std::string value = read_string_from_env(var);
    if (value.empty())
      return default_value;
    return std::stoi(value);
  }

}
