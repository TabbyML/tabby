#include <chrono>
#include <fstream>
#include <iostream>

#include <boost/program_options.hpp>

#include <ctranslate2/translator_pool.h>
#include <ctranslate2/utils.h>

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  po::options_description desc("CTranslate2 translation client");
  desc.add_options()
    ("help", "display available options")
    ("model", po::value<std::string>(),
     "path to the model")
    ("src", po::value<std::string>(),
     "path to the file to translate (read from the standard input if not set)")
    ("tgt", po::value<std::string>(),
     "path to the output file (write to the standard output if not set")
    ("use_vmap", po::bool_switch()->default_value(false),
     "use the vocabulary map included in the model to restrict the target candidates")
    ("batch_size", po::value<size_t>()->default_value(30),
     "batch size")
    ("beam_size", po::value<size_t>()->default_value(5),
     "beam size")
    ("n_best", po::value<size_t>()->default_value(1),
     "n-best list")
    ("with_score", po::bool_switch()->default_value(false),
     "display translation score")
    ("length_penalty", po::value<float>()->default_value(0.6),
     "length penalty")
    ("max_sent_length", po::value<size_t>()->default_value(250),
     "maximum sentence length to produce")
    ("log_throughput", po::bool_switch()->default_value(false),
     "log average tokens per second")
    ("inter_threads", po::value<size_t>()->default_value(1),
     "number of parallel translations")
    ("intra_threads", po::value<size_t>()->default_value(0),
     "number of threads for Intel® MKL (set to 0 to use an automatic value)")
    ("device", po::value<std::string>()->default_value("cpu"),
     "device to use")
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
  ctranslate2::init(intra_threads);

  auto model = ctranslate2::ModelFactory::load(
    ctranslate2::ModelType::Transformer,
    vm["model"].as<std::string>(),
    ctranslate2::str_to_device(vm["device"].as<std::string>()));

  ctranslate2::TranslatorPool translator_pool(inter_threads, model);

  auto options = ctranslate2::TranslationOptions();
  options.beam_size = vm["beam_size"].as<size_t>();
  options.length_penalty = vm["length_penalty"].as<float>();
  options.max_decoding_steps = vm["max_sent_length"].as<size_t>();
  options.num_hypotheses = vm["n_best"].as<size_t>();
  options.use_vmap = vm["use_vmap"].as<bool>();

  std::istream* in = vm.count("src") ? new std::ifstream(vm["src"].as<std::string>()) : &std::cin;
  std::ostream* out = vm.count("tgt") ? new std::ofstream(vm["tgt"].as<std::string>()) : &std::cout;
  bool with_score = vm["with_score"].as<bool>();
  size_t num_tokens = 0;

  auto reader = [](std::istream& in, std::vector<std::string>& tokens) {
    std::string line;
    if (!std::getline(in, line))
      return false;
    std::string token;
    for (size_t i = 0; i < line.length(); ++i) {
      if (line[i] == ' ') {
        if (!token.empty()) {
          tokens.emplace_back(std::move(token));
          token.clear();
        }
      } else {
        token += line[i];
      }
    }
    if (!token.empty())
      tokens.emplace_back(std::move(token));
    return true;
  };

  auto writer = [&num_tokens, &options, &with_score](std::ostream& out,
                                                     const ctranslate2::TranslationResult& result) {
    const auto& hypotheses = result.hypotheses();
    const auto& scores = result.scores();
    num_tokens += hypotheses[0].size();
    for (size_t n = 0; n < hypotheses.size(); ++n) {
      if (with_score)
        out << scores[n] << " ||| ";
      for (size_t i = 0; i < hypotheses[n].size(); ++i) {
        if (i > 0)
          out << ' ';
        out << hypotheses[n][i];
      }
      out << std::endl;
    }
  };

  auto t1 = std::chrono::high_resolution_clock::now();
  translator_pool.consume_stream(*in,
                                 *out,
                                 vm["batch_size"].as<size_t>(),
                                 options,
                                 reader,
                                 writer);
  auto t2 = std::chrono::high_resolution_clock::now();

  if (vm["log_throughput"].as<bool>()) {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cerr << static_cast<double>(num_tokens) / static_cast<double>(duration / 1000) << std::endl;
  }

  return 0;
}
