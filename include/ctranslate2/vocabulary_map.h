#pragma once

#include <istream>
#include <set>
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
    VocabularyMap(std::istream& map_file, const Vocabulary& vocabulary);

    // The returned vector is ordered.
    std::vector<size_t>
    get_candidates(const std::vector<std::vector<std::string>>& source_tokens,
                   const std::vector<std::vector<size_t>>& target_prefix_ids) const;

  private:
    const size_t _vocabulary_size;
    std::set<size_t> _fixed_candidates;
    std::vector<std::unordered_map<std::string, std::vector<size_t>>> _map_rules;
  };

}
