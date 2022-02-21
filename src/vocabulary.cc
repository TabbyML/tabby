#include "ctranslate2/vocabulary.h"

namespace ctranslate2 {

  const std::string Vocabulary::pad_token = "<blank>";
  const std::string Vocabulary::unk_token = "<unk>";
  const std::string Vocabulary::bos_token = "<s>";
  const std::string Vocabulary::eos_token = "</s>";

  Vocabulary::Vocabulary(std::istream& in) {
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
    if (id >= size())
      throw std::invalid_argument("Invalid token ID " + std::to_string(id)
                                  + ": valid IDs are between 0 and " + std::to_string(size() - 1));
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

  template <typename From, typename To>
  To lookup(const Vocabulary& vocab, const From& from);

  template<>
  std::string lookup(const Vocabulary& vocab, const size_t& id) {
    return vocab.to_token(id);
  }

  template<>
  size_t lookup(const Vocabulary& vocab, const std::string& token) {
    return vocab.to_id(token);
  }

  template <typename From, typename To>
  std::vector<std::vector<To>> lookup_batch(const Vocabulary& vocab,
                                            const std::vector<std::vector<From>>& batch_from,
                                            const From* prefix = nullptr,
                                            const From* suffix = nullptr) {
    std::vector<std::vector<To>> batch_to;
    batch_to.reserve(batch_from.size());

    const size_t length_increment = size_t(bool(prefix)) + size_t(bool(suffix));

    for (const auto& from : batch_from) {
      std::vector<To> to;
      to.reserve(from.size() + length_increment);

      if (prefix)
        to.emplace_back(lookup<From, To>(vocab, *prefix));
      for (const auto& element : from)
        to.emplace_back(lookup<From, To>(vocab, element));
      if (suffix)
        to.emplace_back(lookup<From, To>(vocab, *suffix));

      batch_to.emplace_back(std::move(to));
    }

    return batch_to;
  }

  std::vector<std::vector<std::string>>
  Vocabulary::to_tokens(const std::vector<std::vector<size_t>>& batch_ids) const {
    return lookup_batch<size_t, std::string>(*this, batch_ids);
  }

  std::vector<std::vector<size_t>>
  Vocabulary::to_ids(const std::vector<std::vector<std::string>>& batch_tokens,
                     const bool add_bos,
                     const bool add_eos) const {
    return lookup_batch<std::string, size_t>(*this,
                                             batch_tokens,
                                             add_bos ? &bos_token : nullptr,
                                             add_eos ? &eos_token : nullptr);
  }

}
