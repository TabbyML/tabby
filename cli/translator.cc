#include <fstream>
#include <iostream>

#include <cxxopts.hpp>

#include <ctranslate2/translator.h>
#include <ctranslate2/utils.h>
#include <ctranslate2/random.h>
#include <ctranslate2/devices.h>
#include <ctranslate2/profiler.h>

int main(int argc, char* argv[]) {
  cxxopts::Options cmd_options("ct2-translator", "CTranslate2 translator client");
  cmd_options.custom_help("--model <directory> [OPTIONS]");

  cmd_options.add_options("General")
    ("h,help", "Display available options.")
    ("task", "Task to run: translate, score.",
     cxxopts::value<std::string>()->default_value("translate"))
    ("seed", "Seed value of the random generators.",
     cxxopts::value<unsigned int>()->default_value("0"))
    ("log_throughput", "Log average tokens per second at the end of the translation.",
     cxxopts::value<bool>()->default_value("false"))
    ("log_profiling", "Log execution profiling.",
     cxxopts::value<bool>()->default_value("false"))
    ;

  cmd_options.add_options("Device")
    ("inter_threads", "Maximum number of CPU translations to run in parallel.",
     cxxopts::value<size_t>()->default_value("1"))
    ("intra_threads", "Number of computation threads (set to 0 to use the default value).",
     cxxopts::value<size_t>()->default_value("0"))
    ("device", "Device to use (can be cpu, cuda, auto).",
     cxxopts::value<std::string>()->default_value("cpu"))
    ("device_index", "Comma-separated list of device IDs to use.",
     cxxopts::value<std::vector<int>>()->default_value("0"))
    ("cpu_core_offset", "Pin worker threads to CPU cores starting from this offset.",
     cxxopts::value<int>()->default_value("-1"))
    ;

  cmd_options.add_options("Model")
    ("model", "Path to the CTranslate2 model directory.", cxxopts::value<std::string>())
    ("compute_type", "The type used for computation: default, auto, float32, float16, bfloat16, int16, int8, int8_float32, int8_float16, or int8_bfloat16",
     cxxopts::value<std::string>()->default_value("default"))
    ("cuda_compute_type", "Computation type on CUDA devices (overrides compute_type)",
     cxxopts::value<std::string>())
    ("cpu_compute_type", "Computation type on CPU devices (overrides compute_type)",
     cxxopts::value<std::string>())
    ;

  cmd_options.add_options("Data")
    ("src", "Path to the source file (read from the standard input if not set).",
     cxxopts::value<std::string>())
    ("tgt", "Path to the target file.",
     cxxopts::value<std::string>())
    ("out", "Path to the output file (write to the standard output if not set).",
     cxxopts::value<std::string>())
    ("batch_size", "Size of the batch to forward into the model at once.",
     cxxopts::value<size_t>()->default_value("32"))
    ("read_batch_size", "Size of the batch to read at once (defaults to batch_size).",
     cxxopts::value<size_t>()->default_value("0"))
    ("max_queued_batches", "Maximum number of batches to load in advance (set -1 for unlimited, 0 for an automatic value).",
     cxxopts::value<long>()->default_value("0"))
    ("batch_type", "Batch type (can be examples, tokens).",
     cxxopts::value<std::string>()->default_value("examples"))
    ("max_input_length", "Truncate inputs after this many tokens (set 0 to disable).",
     cxxopts::value<size_t>()->default_value("1024"))
    ;

  cmd_options.add_options("Translation")
    ("use_vmap", "Use the vocabulary map included in the model to restrict the target candidates.",
     cxxopts::value<bool>()->default_value("false"))
    ("beam_size", "Beam search size (set 1 for greedy decoding).",
     cxxopts::value<size_t>()->default_value("2"))
    ("patience", "Beam search patience factor.",
     cxxopts::value<float>()->default_value("1"))
    ("sampling_topk", "Sample randomly from the top K candidates.",
     cxxopts::value<size_t>()->default_value("1"))
    ("sampling_topp", "Keep the most probable tokens whose cumulative probability exceeds this value.",
     cxxopts::value<float>()->default_value("1"))
    ("sampling_temperature", "Sampling temperature.",
     cxxopts::value<float>()->default_value("1"))
    ("n_best", "Also output the n-best hypotheses.",
     cxxopts::value<size_t>()->default_value("1"))
    ("with_score", "Also output the translation scores.",
     cxxopts::value<bool>()->default_value("false"))
    ("length_penalty", "Exponential penalty applied to the length during beam search.",
     cxxopts::value<float>()->default_value("1"))
    ("coverage_penalty", "Coverage penalty weight applied during beam search.",
     cxxopts::value<float>()->default_value("0"))
    ("repetition_penalty", "Penalty applied to the score of previously generated tokens (set > 1 to penalize)",
     cxxopts::value<float>()->default_value("1"))
    ("no_repeat_ngram_size", "Prevent repetitions of ngrams with this size (set 0 to disable)",
     cxxopts::value<size_t>()->default_value("0"))
    ("disable_unk", "Disable the generation of the unknown token",
     cxxopts::value<bool>()->default_value("false"))
    ("suppress_sequences", "Disable the generation of some sequences of tokens (sequences are delimited with a comma and tokens with an escaped space)",
     cxxopts::value<std::vector<std::string>>()->default_value(""))
    ("end_token", "Stop the decoding on this token (defaults to the model EOS token).",
     cxxopts::value<std::string>()->default_value(""))
    ("prefix_bias_beta", "Parameter for biasing translations towards given prefix",
     cxxopts::value<float>()->default_value("0"))
    ("max_decoding_length", "Maximum sentence length to generate.",
     cxxopts::value<size_t>()->default_value("256"))
    ("min_decoding_length", "Minimum sentence length to generate.",
     cxxopts::value<size_t>()->default_value("1"))
    ("replace_unknowns", "Replace unknown target tokens by the original source token with the highest attention.",
     cxxopts::value<bool>()->default_value("false"))
    ;

  cmd_options.add_options("Scoring")
    ("with_tokens_score", "Also output the token-level scores.",
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
  if (args.count("seed") != 0)
    ctranslate2::set_random_seed(args["seed"].as<unsigned int>());

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

  ctranslate2::ReplicaPoolConfig pool_config;
  pool_config.num_threads_per_replica = intra_threads;
  pool_config.max_queued_batches = args["max_queued_batches"].as<long>();
  pool_config.cpu_core_offset = args["cpu_core_offset"].as<int>();

  ctranslate2::models::ModelLoader model_loader(args["model"].as<std::string>());
  model_loader.device = device;
  model_loader.device_indices = args["device_index"].as<std::vector<int>>();
  model_loader.compute_type = compute_type;
  model_loader.num_replicas_per_device = inter_threads;

  ctranslate2::Translator translator_pool(model_loader, pool_config);

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
    ctranslate2::init_profiling(device, translator_pool.num_replicas());

  const auto task = args["task"].as<std::string>();
  const auto max_batch_size = args["batch_size"].as<size_t>();
  const auto read_batch_size = args["read_batch_size"].as<size_t>();
  const auto batch_type = ctranslate2::str_to_batch_type(args["batch_type"].as<std::string>());
  ctranslate2::ExecutionStats stats;

  if (task == "translate") {
    ctranslate2::TranslationOptions options;
    options.beam_size = args["beam_size"].as<size_t>();
    options.patience = args["patience"].as<float>();
    options.length_penalty = args["length_penalty"].as<float>();
    options.coverage_penalty = args["coverage_penalty"].as<float>();
    options.repetition_penalty = args["repetition_penalty"].as<float>();
    options.no_repeat_ngram_size = args["no_repeat_ngram_size"].as<size_t>();
    options.disable_unk = args["disable_unk"].as<bool>();
    options.prefix_bias_beta = args["prefix_bias_beta"].as<float>();
    options.sampling_topk = args["sampling_topk"].as<size_t>();
    options.sampling_topp = args["sampling_topp"].as<float>();
    options.sampling_temperature = args["sampling_temperature"].as<float>();
    options.max_input_length = args["max_input_length"].as<size_t>();
    options.max_decoding_length = args["max_decoding_length"].as<size_t>();
    options.min_decoding_length = args["min_decoding_length"].as<size_t>();
    options.num_hypotheses = args["n_best"].as<size_t>();
    options.use_vmap = args["use_vmap"].as<bool>();
    options.return_scores = args["with_score"].as<bool>();
    options.replace_unknowns = args["replace_unknowns"].as<bool>();
    options.end_token = args["end_token"].as<std::string>();

    for (const auto& sequence : args["suppress_sequences"].as<std::vector<std::string>>()) {
      if (sequence.empty())
        continue;
      options.suppress_sequences.emplace_back(ctranslate2::split_tokens(sequence));
    }

    stats = translator_pool.translate_text_file(*source,
                                                *output,
                                                options,
                                                max_batch_size,
                                                read_batch_size,
                                                batch_type,
                                                args["with_score"].as<bool>(),
                                                target);
  } else if (task == "score") {
    if (source == &std::cin || !target)
      throw std::invalid_argument("Score task requires both arguments --src and --tgt to be set");

    ctranslate2::ScoringOptions options;
    options.max_input_length = args["max_input_length"].as<size_t>();
    stats = translator_pool.score_text_file(*source,
                                            *target,
                                            *output,
                                            options,
                                            max_batch_size,
                                            read_batch_size,
                                            batch_type,
                                            args["with_tokens_score"].as<bool>());
  } else {
    throw std::invalid_argument("Invalid task: " + task);
  }

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
