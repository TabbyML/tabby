#include "opennmt/vocabulary_map.h"

#include <fstream>

namespace opennmt {

  VocabularyMap::VocabularyMap(const std::string& map_path, const Vocabulary& vocabulary) {
    std::ifstream map_file(map_path);
    if (!map_file.is_open())
      throw std::invalid_argument("Unable to open dictionary vocab mapping file `" + map_path + "`");

    _static_ids.insert(vocabulary.to_id("<unk>"));
    _static_ids.insert(vocabulary.to_id("<s>"));
    _static_ids.insert(vocabulary.to_id("</s>"));
    _static_ids.insert(vocabulary.to_id("<blank>"));

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
  }

  void VocabularyMap::get_candidates(const std::vector<std::string>& tokens,
                                     std::set<size_t>& candidates) const {
    candidates.insert(_static_ids.begin(), _static_ids.end());

    auto it = _map_rules[0].find("");
    if (it != _map_rules[0].end()) {
      for (const auto& v : it->second)
        candidates.insert(v);
    }

    for (size_t i = 0; i < tokens.size(); i++) {
      std::string token = tokens[i];
      size_t h = 0;
      do {
        if (h > 0) {
          if (i + h >= tokens.size())
            break;
          token += " " + tokens[i + h];
        }
        auto it = _map_rules[h].find(token);
        if (it != _map_rules[h].end()) {
          for (const auto& v : it->second)
            candidates.insert(v);
        }
        h++;
      } while (h < _map_rules.size());
    }
  }

}
