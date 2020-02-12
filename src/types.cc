#include "ctranslate2/types.h"

#include <map>

namespace ctranslate2 {

  static const std::map<DataType, std::string> dtype_names = {
    {DataType::FLOAT, "float"},
    {DataType::INT8, "int8"},
    {DataType::INT16, "int16"},
    {DataType::INT32, "int32"}
  };

  const std::string& dtype_name(DataType type) {
    return dtype_names.at(type);
  }

}
