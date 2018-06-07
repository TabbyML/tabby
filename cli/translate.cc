#include <iostream>
#include <fstream>
#include <chrono>

#include "opennmt/translator.h"
#include "opennmt/transformer.h"

static void translate(opennmt::Translator& translator,
                      const std::vector<std::vector<std::string>>& batch_tokens) {
  auto result = translator.translate_batch(batch_tokens);
  for (size_t i = 0; i < result.size(); ++i) {
    for (size_t t = 0; t < result[i].size(); ++t) {
      if (t > 0)
        std::cout << " ";
      std::cout << result[i][t];
    }
    std::cout << std::endl;
  }
}

int main(int argc, char* argv[]) {
  size_t max_batch_size = argc > 1 ? std::stoi(argv[1]) : 1;
  size_t beam_size = argc > 2 ? std::stoi(argv[2]) : 1;

  std::string model_path = "/home/klein/dev/ctransformer/model.bin";
  std::string vocabulary_path = "/home/klein/data/wmt-ende/wmtende.vocab";
  opennmt::TransformerModel model(model_path, vocabulary_path);
  opennmt::Translator translator(model, 200, beam_size, 0.6, "");

  std::ifstream text_file("/home/klein/data/wmt-ende/valid.en.500");
  std::vector<std::vector<std::string> > input_tokens;
  std::string line;
  size_t num_tokens = 0;

  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

  while (std::getline(text_file, line)) {
    input_tokens.emplace_back();
    std::string token;
    for (size_t i = 0; i < line.length(); ++i) {
      if (line[i] == ' ') {
        if (!token.empty()) {
          input_tokens.back().push_back(token);
          token.clear();
        }
      } else {
        token += line[i];
      }
    }
    if (!token.empty()) {
      input_tokens.back().push_back(token);
      token.clear();
    }
    num_tokens += input_tokens.back().size();

    if (input_tokens.size() == max_batch_size) {
      translate(translator, input_tokens);
      input_tokens.clear();
    }
  }

  if (!input_tokens.empty()) {
    translate(translator, input_tokens);
  }

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  std::cerr << static_cast<double>(num_tokens) / static_cast<double>(duration / 1000) << std::endl;
  return 0;
}
