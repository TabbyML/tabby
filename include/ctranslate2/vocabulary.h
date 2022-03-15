#pragma once

#include <istream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ctranslate2 {

  // Implements a standard indexed vocabulary.
  class Vocabulary
  {
  public:
    static const std::string pad_token;
    static const std::string unk_token;
    static const std::string bos_token;
    static const std::string eos_token;

    Vocabulary(std::istream& in);

    const std::string& to_token(size_t id) const;
    size_t to_id(const std::string& token) const;
    size_t size() const;

    // Helper methods to lookup a batch of tokens or ids.
    std::vector<std::vector<std::string>>
    to_tokens(const std::vector<std::vector<size_t>>& batch_ids) const;
    std::vector<std::vector<size_t>>
    to_ids(const std::vector<std::vector<std::string>>& batch_tokens,
           const bool add_bos = false,
           const bool add_eos = false) const;
    std::vector<std::vector<size_t>>
    to_ids(const std::vector<std::vector<std::string>>& batch_tokens,
           const std::string* prefix,
           const std::string* suffix) const;

  private:
    std::vector<const std::string*> _id_to_token;
    std::unordered_map<std::string, size_t> _token_to_id;
  };

}
