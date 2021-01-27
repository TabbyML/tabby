#include <fstream>
#include <iostream>

#include <cxxopts.hpp>

#include <ctranslate2/translator_pool.h>
#include <ctranslate2/utils.h>
#include <ctranslate2/devices.h>
#include <ctranslate2/profiler.h>

int main(int argc, char* argv[]) {
  cxxopts::Options cmd_options("translate", "CTranslate2 translation client");
  cmd_options.add_options()
    ("h,help", "Display available options.")
    ("model", "Path to the CTranslate2 model directory.", cxxopts::value<std::string>())
    ("compute_type", "The type used for computation: default, float, float16, int16, or int8",
     cxxopts::value<std::string>()->default_value("default"))
    ("cuda_compute_type", "Computation type on CUDA devices (overrides compute_type)",
     cxxopts::value<std::string>())
    ("cpu_compute_type", "Computation type on CPU devices (overrides compute_type)",
     cxxopts::value<std::string>())
    ("src", "Path to the source file (read from the standard input if not set).",
     cxxopts::value<std::string>())
    ("tgt", "Path to the target file.",
     cxxopts::value<std::string>())
    ("out", "Path to the output file (write to the standard output if not set).",
     cxxopts::value<std::string>())
    ("use_vmap", "Use the vocabulary map included in the model to restrict the target candidates.",
     cxxopts::value<bool>()->default_value("false"))
    ("batch_size", "Size of the batch to forward into the model at once.",
     cxxopts::value<size_t>()->default_value("32"))
    ("read_batch_size", "Size of the batch to read at once (defaults to batch_size).",
     cxxopts::value<size_t>()->default_value("0"))
    ("batch_type", "Batch type (can be examples, tokens).",
     cxxopts::value<std::string>()->default_value("examples"))
    ("beam_size", "Beam search size (set 1 for greedy decoding).",
     cxxopts::value<size_t>()->default_value("2"))
    ("sampling_topk", "Sample randomly from the top K candidates.",
     cxxopts::value<size_t>()->default_value("1"))
    ("sampling_temperature", "Sampling temperature.",
     cxxopts::value<float>()->default_value("1"))
    ("n_best", "Also output the n-best hypotheses.",
     cxxopts::value<size_t>()->default_value("1"))
    ("with_score", "Also output translation scores.",
     cxxopts::value<bool>()->default_value("false"))
    ("length_penalty", "Length penalty to apply during beam search",
     cxxopts::value<float>()->default_value("0"))
    ("coverage_penalty", "Coverage penalty to apply during beam search",
     cxxopts::value<float>()->default_value("0"))
    ("max_sent_length", "Maximum sentence length to produce.",
     cxxopts::value<size_t>()->default_value("250"))
    ("min_sent_length", "Minimum sentence length to produce.",
     cxxopts::value<size_t>()->default_value("1"))
    ("log_throughput", "Log average tokens per second at the end of the translation.",
     cxxopts::value<bool>()->default_value("false"))
    ("log_profiling", "Log execution profiling.",
     cxxopts::value<bool>()->default_value("false"))
    ("inter_threads", "Maximum number of CPU translations to run in parallel.",
     cxxopts::value<size_t>()->default_value("1"))
    ("intra_threads", "Number of OpenMP threads (set to 0 to use the default value).",
     cxxopts::value<size_t>()->default_value("4"))
    ("device", "Device to use (can be cpu, cuda, auto).",
     cxxopts::value<std::string>()->default_value("cpu"))
    ("device_index", "Comma-separated list of device IDs to use.",
     cxxopts::value<std::vector<int>>()->default_value("0"))
    ("replace_unknowns", "Replace unknown target tokens by the original source token with the highest attention.",
     cxxopts::value<bool>()->default_value("false"))
    ;

  auto args = cmd_options.parse(argc, argv);

  if (args.count("help")) {
    std::cerr << cmd_options.help() << std::endl;
    return 0;
  }
  if (!args.count("model")) {
    throw std::invalid_argument("Option --model is required to run translation");
  }

  size_t inter_threads = args["inter_threads"].as<size_t>();
  size_t intra_threads = args["intra_threads"].as<size_t>();

  const auto device = ctranslate2::str_to_device(args["device"].as<std::string>());
  auto compute_type = ctranslate2::str_to_compute_type(args["compute_type"].as<std::string>());
  switch (device) {
  case ctranslate2::Device::CPU:
    if (args.count("cpu_compute_type"))
      compute_type = ctranslate2::str_to_compute_type(args["cpu_compute_type"].as<std::string>());
    break;
  case ctranslate2::Device::CUDA:
    if (args.count("cuda_compute_type"))
      compute_type = ctranslate2::str_to_compute_type(args["cuda_compute_type"].as<std::string>());
    break;
  };

  ctranslate2::TranslatorPool translator_pool(inter_threads,
                                              intra_threads,
                                              args["model"].as<std::string>(),
                                              device,
                                              args["device_index"].as<std::vector<int>>(),
                                              compute_type);

  auto options = ctranslate2::TranslationOptions();
  options.max_batch_size = args["batch_size"].as<size_t>();
  options.batch_type = ctranslate2::str_to_batch_type(args["batch_type"].as<std::string>());
  options.beam_size = args["beam_size"].as<size_t>();
  options.length_penalty = args["length_penalty"].as<float>();
  options.coverage_penalty = args["coverage_penalty"].as<float>();
  options.sampling_topk = args["sampling_topk"].as<size_t>();
  options.sampling_temperature = args["sampling_temperature"].as<float>();
  options.max_decoding_length = args["max_sent_length"].as<size_t>();
  options.min_decoding_length = args["min_sent_length"].as<size_t>();
  options.num_hypotheses = args["n_best"].as<size_t>();
  options.use_vmap = args["use_vmap"].as<bool>();
  options.return_scores = args["with_score"].as<bool>();
  options.replace_unknowns = args["replace_unknowns"].as<bool>();

  std::istream* source = &std::cin;
  std::istream* target = nullptr;
  std::ostream* output = &std::cout;
  if (args.count("src")) {
    auto path = args["src"].as<std::string>();
    auto src_file = new std::ifstream(path);
    if (!src_file->is_open())
      throw std::runtime_error("Unable to open source file " + path);
    source = src_file;
  }
  if (args.count("tgt")) {
    auto path = args["tgt"].as<std::string>();
    auto tgt_file = new std::ifstream(path);
    if (!tgt_file->is_open())
      throw std::runtime_error("Unable to open target file " + path);
    target = tgt_file;
  }
  if (args.count("out")) {
    output = new std::ofstream(args["out"].as<std::string>());
  }

  auto log_profiling = args["log_profiling"].as<bool>();
  if (log_profiling)
    ctranslate2::init_profiling(device, translator_pool.num_translators());
  auto read_batch_size = args["read_batch_size"].as<size_t>();
  if (read_batch_size == 0)
    read_batch_size = options.max_batch_size;
  const ctranslate2::TranslationStats stats = translator_pool.consume_text_file(
    *source,
    *output,
    read_batch_size,
    options,
    args["with_score"].as<bool>(),
    target);
  if (log_profiling)
    ctranslate2::dump_profiling(std::cerr);

  if (source != &std::cin)
    delete source;
  if (target)
    delete target;
  if (output != &std::cout)
    delete output;

  if (args["log_throughput"].as<bool>()) {
    std::cerr << static_cast<double>(stats.num_tokens) / (stats.total_time_in_ms / 1000) << std::endl;
  }

  return 0;
}
