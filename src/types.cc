#include "ctranslate2/types.h"

#include <map>

namespace ctranslate2 {

  static const std::map<DataType, std::string> dtype_names = {
    {DataType::DT_FLOAT, "float"},
    {DataType::DT_INT8, "int8"},
    {DataType::DT_INT16, "int16"},
    {DataType::DT_INT32, "int32"}
  };

  const std::string& dtype_name(DataType type) {
    return dtype_names.at(type);
  }

}
