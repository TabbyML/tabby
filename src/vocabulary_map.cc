#include "ctranslate2/vocabulary_map.h"

#include <fstream>

namespace ctranslate2 {

  VocabularyMap::VocabularyMap(const std::string& map_path, const Vocabulary& vocabulary)
    : _vocabulary_size(vocabulary.size()) {
    std::ifstream map_file(map_path);
    if (!map_file.is_open())
      return;

    std::string line;
    while (std::getline(map_file, line)) {
      std::string token;
      std::string key;
      std::vector<size_t> values;
      bool target = false;
      size_t ngram = 1;

      for (size_t i = 0; i < line.length(); ++i) {
        if (line[i] == '\t') {
          target = true;
          std::swap(key, token);
        } else if (line[i] == ' ') {
          if (target) {
            values.push_back(vocabulary.to_id(token));
            token.clear();
          } else {
            token += line[i];
            ++ngram;
          }
        } else
          token += line[i];
      }

      if (!token.empty())
        values.push_back(vocabulary.to_id(token));

      if (ngram > _map_rules.size())
        _map_rules.resize(ngram);

      _map_rules[ngram - 1][key] = values;
    }

    _fixed_candidates.insert(vocabulary.to_id(Vocabulary::unk_token));
    _fixed_candidates.insert(vocabulary.to_id(Vocabulary::bos_token));
    _fixed_candidates.insert(vocabulary.to_id(Vocabulary::eos_token));
    _fixed_candidates.insert(vocabulary.to_id(Vocabulary::pad_token));

    // The field marked by the empty string are common tokens that are always candidates.
    auto it = _map_rules[0].find("");
    if (it != _map_rules[0].end())
      _fixed_candidates.insert(it->second.begin(), it->second.end());
  }

  bool VocabularyMap::empty() const {
    return _map_rules.empty();
  }

  std::vector<size_t>
  VocabularyMap::get_candidates(const std::vector<std::vector<std::string>>& batch_tokens,
                                const size_t multiple_of) const {
    std::set<size_t> candidates = _fixed_candidates;
    std::string accu;
    for (const auto& tokens : batch_tokens) {
      for (size_t i = 0; i < tokens.size(); i++) {
        accu.clear();
        for (size_t h = 0; h < _map_rules.size() && i + h < tokens.size(); ++h) {
          if (h > 0)
            accu += ' ';
          accu += tokens[i + h];
          auto it = _map_rules[h].find(accu);
          if (it != _map_rules[h].end())
            candidates.insert(it->second.begin(), it->second.end());
        }
      }
    }

    std::vector<size_t> ids(candidates.begin(), candidates.end());
    const size_t num_candidates = ids.size();
    const size_t padding_size = multiple_of - (num_candidates % multiple_of);
    if (padding_size != multiple_of && num_candidates + padding_size <= _vocabulary_size) {
      std::vector<size_t> padding(padding_size, 0);
      ids.insert(ids.begin(), padding.begin(), padding.end());
    }

    return ids;
  }

}
