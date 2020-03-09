#include "ctranslate2/vocabulary.h"

#include <fstream>

namespace ctranslate2 {

  const std::string Vocabulary::pad_token = "<blank>";
  const std::string Vocabulary::unk_token = "<unk>";
  const std::string Vocabulary::bos_token = "<s>";
  const std::string Vocabulary::eos_token = "</s>";

  // Most vocabularies are typically smaller than this size. This hint is used to reserve
  // memory upfront and avoid multiple re-allocations and copies.
  constexpr size_t VOCABULARY_SIZE_HINT = 50000;

  Vocabulary::Vocabulary(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open())
      throw std::invalid_argument("Unable to open the vocabulary file `" + path + "`");
    _token_to_id.reserve(VOCABULARY_SIZE_HINT);
    _id_to_token.reserve(VOCABULARY_SIZE_HINT);
    std::string line;
    while (std::getline(in, line)) {
      const auto result = _token_to_id.emplace(std::move(line), _id_to_token.size());
      _id_to_token.emplace_back(&result.first->first);
    }
    // Append the unknown token if not found in the vocabulary file.
    if (_token_to_id.find(unk_token) == _token_to_id.end()) {
      _token_to_id.emplace(unk_token, _id_to_token.size());
      _id_to_token.emplace_back(&unk_token);
    }
  }

  const std::string& Vocabulary::to_token(size_t id) const {
    return *_id_to_token[id];
  }

  size_t Vocabulary::to_id(const std::string& token) const {
    auto it = _token_to_id.find(token);
    if (it == _token_to_id.end())
      return _token_to_id.at(unk_token);
    return it->second;
  }

  size_t Vocabulary::size() const {
    return _id_to_token.size();
  }

}
