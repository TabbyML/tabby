#include <iostream>
#include <fstream>
#include <chrono>

#include "opennmt/translator.h"
#include "opennmt/transformer.h"

int main(int argc, char* argv[]) {
  opennmt::Model model("/home/klein/dev/ctransformer/model.bin");
  opennmt::Vocabulary vocabulary("/home/klein/data/wmt-ende/wmtende.vocab");

  opennmt::TransformerEncoder encoder(model, "transformer/encoder");
  opennmt::TransformerDecoder decoder(model, "transformer/decoder");

  std::ifstream text_file("/home/klein/data/wmt-ende/valid.en");
  std::vector<std::vector<std::string> > input_tokens;
  std::string line;

  size_t max_batch_size = argc > 1 ? std::stoi(argv[1]) : 1;
  size_t beam_size = argc > 2 ? std::stoi(argv[2]) : 1;
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
      opennmt::translate(input_tokens, vocabulary, encoder, decoder, beam_size);
      input_tokens.clear();
    }
  }

  if (!input_tokens.empty()) {
    opennmt::translate(input_tokens, vocabulary, encoder, decoder, beam_size);
  }

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  std::cerr << static_cast<double>(num_tokens) / static_cast<double>(duration / 1000) << std::endl;
  return 0;
}
