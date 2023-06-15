#pragma once

#include <istream>
#include <string>
#include <unordered_map>
#include <vector>

namespace ctranslate2 {

  struct VocabularyInfo {
    std::string unk_token = "<unk>";
    std::string bos_token = "<s>";
    std::string eos_token = "</s>";
  };

  // Implements a standard indexed vocabulary.
  class Vocabulary
  {
  public:
    static Vocabulary from_text_file(std::istream& in, VocabularyInfo info = VocabularyInfo());
    static Vocabulary from_json_file(std::istream& in, VocabularyInfo info = VocabularyInfo());

    Vocabulary(std::vector<std::string> tokens, VocabularyInfo info = VocabularyInfo());

    bool contains(const std::string& token) const;
    const std::string& to_token(size_t id) const;
    size_t to_id(const std::string& token, const bool allow_unk = true) const;
    size_t size() const;

    // Helper methods to lookup a batch of tokens or ids.
    std::vector<std::vector<std::string>>
    to_tokens(const std::vector<std::vector<size_t>>& batch_ids) const;
    std::vector<std::vector<size_t>>
    to_ids(const std::vector<std::vector<std::string>>& batch_tokens,
           const size_t max_length = 0,
           const bool add_bos = false,
           const bool add_eos = false) const;
    std::vector<std::vector<size_t>>
    to_ids(const std::vector<std::vector<std::string>>& batch_tokens,
           const size_t max_length,
           const std::string* prefix,
           const std::string* suffix,
           const bool allow_unk = true) const;

    const std::string& unk_token() const {
      return _info.unk_token;
    }
    const std::string& bos_token() const {
      return _info.bos_token;
    }
    const std::string& eos_token() const {
      return _info.eos_token;
    }

    size_t unk_id() const {
      return to_id(_info.unk_token);
    }
    size_t bos_id() const {
      return to_id(_info.bos_token);
    }
    size_t eos_id() const {
      return to_id(_info.eos_token);
    }

  private:
    std::vector<const std::string*> _id_to_token;
    std::unordered_map<std::string, size_t> _token_to_id;
    const VocabularyInfo _info;

    void add_token(std::string token);
  };

}
