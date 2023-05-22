#include "ctranslate2/filesystem.h"

#include <stdexcept>

#ifdef _WIN32
#  include <windows.h>
#endif

namespace ctranslate2 {

#ifdef _WIN32
  static std::wstring convert_to_wstring(const std::string& str) {
    int count = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
    std::wstring wstr(count, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], count);
    return wstr;
  }
#endif

  template <typename Stream>
  static Stream open_file(const std::string& path,
                          std::ios_base::openmode mode,
                          bool check) {
#ifdef _WIN32
    const std::wstring wpath = convert_to_wstring(path);
    Stream stream(wpath.c_str(), mode);
#else
    Stream stream(path, mode);
#endif

    if (check && !stream)
      throw std::runtime_error("Failed to open file: " + path);

    return stream;
  }

  std::ifstream open_file_read(const std::string& path,
                               std::ios_base::openmode mode,
                               bool check) {
    return open_file<std::ifstream>(path, mode, check);
  }

  std::ofstream open_file_write(const std::string& path,
                                std::ios_base::openmode mode,
                                bool check) {
    return open_file<std::ofstream>(path, mode, check);
  }

}
