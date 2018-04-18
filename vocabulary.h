#pragma once

#include <string>
#include <unordered_map>
#include <vector>

class Vocabulary
{
public:
  static const std::string unk_token;

  Vocabulary(const char* path) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
      _token_to_id.emplace(line, _id_to_token.size());
      _id_to_token.push_back(line);
    }
    _token_to_id.emplace(unk_token, _id_to_token.size());
    _id_to_token.push_back(unk_token);

  }

  const std::string& to_token(size_t id) const {
    return _id_to_token[id];
  }
  size_t to_id(const std::string& token) const {
    auto it = _token_to_id.find(token);
    if (it == _token_to_id.end())
      return _token_to_id.at(unk_token);
    return it->second;
  }
  size_t size() const {
    return _id_to_token.size();
  }

private:
  std::vector<std::string> _id_to_token;
  std::unordered_map<std::string, size_t> _token_to_id;
};

const std::string Vocabulary::unk_token = "<unk>";
