#include <chrono>
#include <fstream>
#include <iostream>

#include <boost/program_options.hpp>

#include <ctranslate2/translator_pool.h>
#include <ctranslate2/utils.h>
#include <ctranslate2/devices.h>
#include <ctranslate2/profiler.h>

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  po::options_description desc("CTranslate2 translation client");
  desc.add_options()
    ("help", "Display available options.")
    ("model", po::value<std::string>(),
     "Path to the CTranslate2 model directory.")
    ("compute_type", po::value<std::string>()->default_value("default"),
     "Force the model type as \"float\", \"int16\" or \"int8\"")
    ("src", po::value<std::string>(),
     "Path to the file to translate (read from the standard input if not set).")
    ("tgt", po::value<std::string>(),
     "Path to the output file (write to the standard output if not set.")
    ("use_vmap", po::bool_switch()->default_value(false),
     "Use the vocabulary map included in the model to restrict the target candidates.")
    ("batch_size", po::value<size_t>()->default_value(30),
     "Number of sentences to forward into the model at once.")
    ("beam_size", po::value<size_t>()->default_value(5),
     "Beam search size (set 1 for greedy decoding).")
    ("n_best", po::value<size_t>()->default_value(1),
     "Also output the n-best hypotheses.")
    ("with_score", po::bool_switch()->default_value(false),
     "Also output translation scores.")
    ("length_penalty", po::value<float>()->default_value(0),
     "Length penalty to apply during beam search")
    ("max_sent_length", po::value<size_t>()->default_value(250),
     "Maximum sentence length to produce.")
    ("min_sent_length", po::value<size_t>()->default_value(1),
     "Minimum sentence length to produce.")
    ("log_throughput", po::bool_switch()->default_value(false),
     "Log average tokens per second at the end of the translation.")
    ("log_profiling", po::bool_switch()->default_value(false),
     "Log execution profiling.")
    ("inter_threads", po::value<size_t>()->default_value(1),
     "Maximum number of translations to run in parallel.")
    ("intra_threads", po::value<size_t>()->default_value(0),
     "Number of OpenMP threads (set to 0 to use the default value).")
    ("device", po::value<std::string>()->default_value("cpu"),
     "Device to use (can be cpu, cuda, auto).")
    ("device_index", po::value<int>()->default_value(0),
     "Index of the device to use.")
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

  auto model = ctranslate2::models::Model::load(
    vm["model"].as<std::string>(),
    vm["device"].as<std::string>(),
    vm["device_index"].as<int>(),
    vm["compute_type"].as<std::string>());

  ctranslate2::TranslatorPool translator_pool(inter_threads, intra_threads, model);

  auto options = ctranslate2::TranslationOptions();
  options.beam_size = vm["beam_size"].as<size_t>();
  options.length_penalty = vm["length_penalty"].as<float>();
  options.max_decoding_length = vm["max_sent_length"].as<size_t>();
  options.min_decoding_length = vm["min_sent_length"].as<size_t>();
  options.num_hypotheses = vm["n_best"].as<size_t>();
  options.use_vmap = vm["use_vmap"].as<bool>();

  std::istream* in = &std::cin;
  std::ostream* out = &std::cout;
  if (vm.count("src")) {
    auto path = vm["src"].as<std::string>();
    auto src_file = new std::ifstream(path);
    if (!src_file->is_open())
      throw std::runtime_error("Unable to open input file " + path);
    in = src_file;
  }
  if (vm.count("tgt")) {
    out = new std::ofstream(vm["tgt"].as<std::string>());
  }

  auto log_profiling = vm["log_profiling"].as<bool>();
  auto t1 = std::chrono::high_resolution_clock::now();
  if (log_profiling)
    ctranslate2::init_profiling(model->device(), inter_threads);
  auto num_tokens = translator_pool.consume_text_file(*in,
                                                      *out,
                                                      vm["batch_size"].as<size_t>(),
                                                      options,
                                                      vm["with_score"].as<bool>());
  if (log_profiling)
    ctranslate2::dump_profiling(std::cerr);
  auto t2 = std::chrono::high_resolution_clock::now();

  if (in != &std::cin)
    delete in;
  if (out != &std::cout)
    delete out;

  if (vm["log_throughput"].as<bool>()) {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cerr << static_cast<double>(num_tokens) / static_cast<double>(duration / 1000) << std::endl;
  }

  return 0;
}
