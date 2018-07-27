#pragma once

#include <unordered_set>
#include <unordered_map>
#include <string>
#include <vector>

#include "vocabulary.h"

namespace ctranslate2 {

  // This class loads a vocabulary mapping model, a text file associating n-grams
  // with list of possible target candidates:
  //
  //    <n-gram> \t candidate1 candidate2 ... candidateN
  //
  // and provides methods to map input tokens to possible target tokens.
  class VocabularyMap {
  public:
    VocabularyMap(const std::string& map_path, const Vocabulary& vocabulary);

    bool empty() const;

    template <typename T>
    std::vector<T>
    get_candidates(const std::vector<std::vector<std::string>>& batch_tokens) const {
      std::unordered_set<size_t> candidates = _fixed_candidates;
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
      return std::vector<T>(candidates.begin(), candidates.end());
    }

  private:
    std::unordered_set<size_t> _fixed_candidates;
    std::vector<std::unordered_map<std::string, std::vector<size_t>>> _map_rules;
  };

}
