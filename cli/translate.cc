#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <thread>

#include "opennmt/translator.h"
#include "opennmt/transformer.h"

struct Batch {
  std::vector<std::vector<std::string>> tokens;
  size_t id;

  bool empty() const {
    return tokens.empty();
  }
  size_t size() const {
    return tokens.size();
  }
  size_t num_tokens() const {
    size_t num = 0;
    for (const auto& x : tokens)
      num += x.size();
    return num;
  }
};

std::ostream& operator<<(std::ostream& os, const Batch& batch) {
  for (size_t i = 0; i < batch.tokens.size(); ++i) {
    for (size_t t = 0; t < batch.tokens[i].size(); ++t) {
      if (t > 0)
        os << " ";
      os << batch.tokens[i][t];
    }
    os << std::endl;
  }
  return os;
}

class ConcurrentReader {
public:
  ConcurrentReader(std::istream& in)
    : _in(in)
    , _batch_id(0) {
  }

  Batch read(size_t max_batch_size) {
    std::lock_guard<std::mutex> lock(_mutex);

    Batch batch;
    batch.id = ++_batch_id;

    std::string line;
    while (batch.size() < max_batch_size && std::getline(_in, line)) {
      batch.tokens.emplace_back();
      std::string token;
      for (size_t i = 0; i < line.length(); ++i) {
        if (line[i] == ' ') {
          if (!token.empty()) {
            batch.tokens.back().push_back(token);
            token.clear();
          }
        } else {
          token += line[i];
        }
      }
      if (!token.empty()) {
        batch.tokens.back().push_back(token);
        token.clear();
      }
    }

    return batch;
  }

private:
  std::mutex _mutex;
  std::istream& _in;
  size_t _batch_id;
};

class ConcurrentWriter {
public:
  ConcurrentWriter(std::ostream& out)
    : _out(out)
    , _last_batch_id(0) {
  }

  void write(const Batch& batch) {
    std::lock_guard<std::mutex> lock(_mutex);
    size_t batch_id = batch.id;
    if (batch_id == _last_batch_id + 1) {
      Batch v = batch;
      while (batch_id == _last_batch_id + 1) {
        _out << v;
        _last_batch_id = batch_id;
        auto it = _pending_batches.find(_last_batch_id + 1);
        if (it != _pending_batches.end()) {
          v = it->second;
          batch_id = _last_batch_id + 1;
          _pending_batches.erase(it);
        }
      }
    }
    else
      _pending_batches[batch_id] = batch;
  }

private:
  std::mutex _mutex;
  std::ostream& _out;
  size_t _last_batch_id;
  std::map<size_t, Batch> _pending_batches;
};

int main(int argc, char* argv[]) {
  size_t max_batch_size = argc > 1 ? std::stoi(argv[1]) : 1;
  size_t beam_size = argc > 2 ? std::stoi(argv[2]) : 1;
  size_t inter_threads = argc > 3 ? std::stoi(argv[3]) : 1;
  std::string model_path = argc > 4 ? argv[4] : "/home/klein/dev/ctransformer/model.bin";
  std::string vocabulary_path = "/home/klein/data/wmt-ende/wmtende.vocab";
  opennmt::TransformerModel model(model_path, vocabulary_path);

  std::vector<opennmt::Translator> translator_pool;
  translator_pool.emplace_back(model, 200, beam_size, 0.6, "");
  for (size_t i = 1; i < inter_threads; ++i) {
    translator_pool.emplace_back(translator_pool.front());
  }

  std::ifstream text_file("/home/klein/data/wmt-ende/valid.en.500");
  ConcurrentReader reader(text_file);
  ConcurrentWriter writer(std::cout);

  std::vector<std::future<size_t>> futures;

  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

  for (auto& translator : translator_pool) {
    futures.emplace_back(
      std::async(std::launch::async,
                 [&max_batch_size](ConcurrentReader* r,
                                   ConcurrentWriter* w,
                                   opennmt::Translator* t) {
                   size_t num_tokens = 0;
                   while (true) {
                     Batch in_batch = r->read(max_batch_size);
                     if (in_batch.empty())
                       break;
                     Batch out_batch;
                     out_batch.tokens = t->translate_batch(in_batch.tokens);
                     out_batch.id = in_batch.id;
                     num_tokens += out_batch.num_tokens();
                     w->write(out_batch);
                   }
                   return num_tokens;
                 },
                 &reader, &writer, &translator));
  }

  size_t num_tokens = 0;
  for (auto& f : futures) {
    f.wait();
    num_tokens += f.get();
  }

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  std::cerr << static_cast<double>(num_tokens) / static_cast<double>(duration / 1000) << std::endl;
  return 0;
}
