#pragma once

#include <fstream>
#include <string>

namespace ctranslate2 {

  std::ifstream open_file_read(const std::string& path,
                               std::ios_base::openmode mode = std::ios_base::in,
                               bool check = true);

  std::ofstream open_file_write(const std::string& path,
                                std::ios_base::openmode mode = std::ios_base::out,
                                bool check = true);

}
