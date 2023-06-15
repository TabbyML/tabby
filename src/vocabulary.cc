#include "ctranslate2/vocabulary.h"

#include <nlohmann/json.hpp>

#include "ctranslate2/utils.h"

namespace ctranslate2 {

  Vocabulary Vocabulary::from_text_file(std::istream& in, VocabularyInfo info) {
    std::vector<std::string> tokens;
    std::string line;

    // Some vocabularies contain tokens ending with a carriage return. In that case
    // the carriage return should be kept so we check that all lines end with a carriage
    // return before removing it.
    bool remove_carriage_return = true;

    while (ctranslate2::getline(in, line, /*remove_carriage_return=*/false)) {
      if (line.empty() || line.back() != '\r')
        remove_carriage_return = false;
      tokens.emplace_back(std::move(line));
    }

    if (remove_carriage_return) {
      for (auto& token : tokens)
        token.pop_back();
    }

    return Vocabulary(std::move(tokens), std::move(info));
  }

  Vocabulary Vocabulary::from_json_file(std::istream& in, VocabularyInfo info) {
    return Vocabulary(nlohmann::json::parse(in).get<std::vector<std::string>>(), std::move(info));
  }

  Vocabulary::Vocabulary(std::vector<std::string> tokens, VocabularyInfo info)
    : _info(std::move(info))
  {
    _token_to_id.reserve(tokens.size());
    _id_to_token.reserve(tokens.size());

    for (auto& token : tokens) {
      add_token(std::move(token));
    }

    // Append the unknown token if not found in the vocabulary file.
    if (!contains(_info.unk_token))
      add_token(_info.unk_token);
  }

  void Vocabulary::add_token(std::string token) {
    const auto result = _token_to_id.emplace(std::move(token), _id_to_token.size());
    _id_to_token.emplace_back(&result.first->first);
  }

  bool Vocabulary::contains(const std::string& token) const {
    return _token_to_id.find(token) != _token_to_id.end();
  }

  const std::string& Vocabulary::to_token(size_t id) const {
    if (id >= size())
      throw std::invalid_argument("Invalid token ID " + std::to_string(id)
                                  + ": valid IDs are between 0 and " + std::to_string(size() - 1));
    return *_id_to_token[id];
  }

  size_t Vocabulary::to_id(const std::string& token, const bool allow_unk) const {
    auto it = _token_to_id.find(token);
    if (it == _token_to_id.end()) {
      if (!allow_unk && token != _info.unk_token)
        throw std::invalid_argument("Token " + token + " is not in the vocabulary");
      return _token_to_id.at(_info.unk_token);
    }
    return it->second;
  }

  size_t Vocabulary::size() const {
    return _id_to_token.size();
  }

  std::vector<std::vector<std::string>>
  Vocabulary::to_tokens(const std::vector<std::vector<size_t>>& batch_ids) const {
    std::vector<std::vector<std::string>> batch_tokens;
    batch_tokens.reserve(batch_ids.size());

    for (const auto& ids : batch_ids) {
      std::vector<std::string> tokens;
      tokens.reserve(ids.size());
      for (const auto id : ids)
        tokens.emplace_back(to_token(id));
      batch_tokens.emplace_back(std::move(tokens));
    }

    return batch_tokens;
  }

  std::vector<std::vector<size_t>>
  Vocabulary::to_ids(const std::vector<std::vector<std::string>>& batch_tokens,
                     const size_t max_length,
                     const bool add_bos,
                     const bool add_eos) const {
    return to_ids(batch_tokens,
                  max_length,
                  add_bos ? &_info.bos_token : nullptr,
                  add_eos ? &_info.eos_token : nullptr);
  }

  std::vector<std::vector<size_t>>
  Vocabulary::to_ids(const std::vector<std::vector<std::string>>& batch_tokens,
                     const size_t max_length,
                     const std::string* prefix,
                     const std::string* suffix,
                     const bool allow_unk) const {
    std::vector<std::vector<size_t>> batch_ids;
    batch_ids.reserve(batch_tokens.size());

    const size_t length_increment = size_t(bool(prefix)) + size_t(bool(suffix));

    for (const auto& tokens : batch_tokens) {
      std::vector<size_t> ids;
      ids.reserve(tokens.size() + length_increment);

      if (prefix)
        ids.emplace_back(to_id(*prefix, allow_unk));
      for (const auto& token : tokens)
        ids.emplace_back(to_id(token, allow_unk));
      if (suffix)
        ids.emplace_back(to_id(*suffix, allow_unk));

      if (max_length > 0 && ids.size() > max_length) {
        // Keep EOS and optional lang code in the last positions.
        const size_t eos = eos_id();
        if (ids[ids.size() - 1] == eos)
          ids[max_length - 1] = eos;
        else if (ids[ids.size() - 2] == eos && max_length >= 2) {
          ids[max_length - 2] = eos;
          ids[max_length - 1] = ids.back();
        }

        ids.resize(max_length);
      }

      batch_ids.emplace_back(std::move(ids));
    }

    return batch_ids;
  }

}
