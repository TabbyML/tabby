#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace ctranslate2 {

  // Implements an standard indexed vocabulary
  class Vocabulary
  {
  public:
    static const std::string pad_token;
    static const std::string unk_token;
    static const std::string bos_token;
    static const std::string eos_token;

    Vocabulary(const std::string& path);

    const std::string& to_token(size_t id) const;
    size_t to_id(const std::string& token) const;
    size_t size() const;

  private:
    std::vector<const std::string*> _id_to_token;
    std::unordered_map<std::string, size_t> _token_to_id;
  };

}
