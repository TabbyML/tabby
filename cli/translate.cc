#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <thread>

#include <boost/program_options.hpp>

#include "opennmt/translator.h"
#include "opennmt/utils.h"

namespace po = boost::program_options;

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
  ConcurrentReader(const std::string& path)
    : _file(path)
    , _in(_file)
    , _batch_id(0) {
    if (!_file.is_open())
      throw std::invalid_argument("Unable to open file " + path);
  }
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
  std::ifstream _file;
  std::istream& _in;
  size_t _batch_id;
};

class ConcurrentWriter {
public:
  ConcurrentWriter(const std::string& path)
    : _file(path)
    , _out(_file)
    , _last_batch_id(0) {
  }
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
  std::ofstream _file;
  std::ostream& _out;
  size_t _last_batch_id;
  std::map<size_t, Batch> _pending_batches;
};

int main(int argc, char* argv[]) {
  po::options_description desc("OpenNMT translator");
  desc.add_options()
    ("help", "display available options")
    ("model", po::value<std::string>(), "path to the model")
    ("src", po::value<std::string>(), "path to the file to translate (read from the standard input if not set)")
    ("tgt", po::value<std::string>(), "path to the output file (write to the standard output if not set")
    ("vocab_mapping", po::value<std::string>()->default_value(""), "path to a vocabulary mapping table")
    ("batch_size", po::value<size_t>()->default_value(30), "batch size")
    ("beam_size", po::value<size_t>()->default_value(5), "beam size")
    ("length_penalty", po::value<float>()->default_value(0.6), "length penalty")
    ("max_sent_length", po::value<size_t>()->default_value(250), "maximum sentence length to produce")
    ("log_throughput", po::bool_switch()->default_value(false), "log average tokens per second")
    ("inter_threads", po::value<size_t>()->default_value(1), "number of inter batch threads")
    ("intra_threads", po::value<size_t>()->default_value(0), "number of intra batch threads (set to 0 to use an automatic value)")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    return 1;
  }
  if (!vm.count("model")) {
    std::cerr << "missing model" << std::endl;
    return 1;
  }

  size_t inter_threads = vm["inter_threads"].as<size_t>();
  size_t intra_threads = vm["intra_threads"].as<size_t>();
  opennmt::init(intra_threads);

  auto model = opennmt::ModelFactory::load(opennmt::ModelType::Transformer,
                                           vm["model"].as<std::string>());

  std::vector<opennmt::Translator> translator_pool;
  translator_pool.emplace_back(model,
                               vm["max_sent_length"].as<size_t>(),
                               vm["beam_size"].as<size_t>(),
                               vm["length_penalty"].as<float>(),
                               vm["vocab_mapping"].as<std::string>());
  for (size_t i = 1; i < inter_threads; ++i) {
    translator_pool.emplace_back(translator_pool.front());
  }

  std::unique_ptr<ConcurrentReader> reader;
  std::unique_ptr<ConcurrentWriter> writer;
  if (vm.count("src"))
    reader.reset(new ConcurrentReader(vm["src"].as<std::string>()));
  else
    reader.reset(new ConcurrentReader(std::cin));
  if (vm.count("tgt"))
    writer.reset(new ConcurrentWriter(vm["tgt"].as<std::string>()));
  else
    writer.reset(new ConcurrentWriter(std::cout));

  size_t max_batch_size = vm["batch_size"].as<size_t>();
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
                 reader.get(), writer.get(), &translator));
  }

  size_t num_tokens = 0;
  for (auto& f : futures) {
    f.wait();
    num_tokens += f.get();
  }

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  if (vm["log_throughput"].as<bool>()) {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cerr << static_cast<double>(num_tokens) / static_cast<double>(duration / 1000) << std::endl;
  }

  return 0;
}
