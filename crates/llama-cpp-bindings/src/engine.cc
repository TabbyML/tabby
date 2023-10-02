#include "engine.h"

#include <functional>
#include <vector>

#include <ggml.h>
#include <llama.h>

namespace llama {
TextInferenceEngine::~TextInferenceEngine() {}

namespace {
static size_t N_BATCH = 512;  // # per batch inference.
static size_t N_CTX = 4096;   // # max kv history.

template<class T>
using owned = std::unique_ptr<T, std::function<void(T*)>>;

class TextInferenceEngineImpl : public TextInferenceEngine {
 public:
  TextInferenceEngineImpl(owned<llama_model> model, owned<llama_context> ctx) :
    model_(std::move(model)),
    ctx_(std::move(ctx)) {
      batch_ = llama_batch_init(N_BATCH, 0);
  }

  void start(rust::Slice<const uint32_t> input_token_ids) override {
    auto* ctx = ctx_.get();
    llama_reset_timings(ctx);
    std::vector<llama_token> tokens_list(input_token_ids.begin(), input_token_ids.end());

    for (size_t i = 0; i < tokens_list.size(); i += N_BATCH) {
      const size_t size = std::min(N_BATCH, tokens_list.size() - i);
      eval(tokens_list.data() + i, size, /* reset = */ i == 0);
    }
  }

  uint32_t step() override {
    const llama_token id = sample();
    eval(const_cast<llama_token*>(&id), 1, /* reset = */ false);
    return id;
  }

  void end() override {
    llama_print_timings(ctx_.get());
  }

  uint32_t eos_token() const override {
    return llama_token_eos(ctx_.get());
  }

 private:
  uint32_t sample() const {
    auto* ctx = ctx_.get();

    auto logits = llama_get_logits_ith(ctx, batch_.n_tokens - 1);
    auto n_vocab = llama_n_vocab(llama_get_model(ctx));

    // Greedy sampling (always select the highest logit).
    return std::distance(logits, std::max_element(logits, logits + n_vocab));
  }

  void eval(llama_token* data, size_t size, bool reset) {
    if (reset) {
      n_past_ = 0;
    }

    batch_.n_tokens = size;
    for (size_t i = 0; i < size; ++i) {
      batch_.token[i] = data[i];
      batch_.pos[i] = n_past_ + i;
      batch_.seq_id[i] = 0;
      batch_.logits[i] = false;
    }
    batch_.logits[size - 1] = true;

    auto* ctx = ctx_.get();
    llama_kv_cache_tokens_rm(ctx, n_past_, -1);
    if (llama_decode(ctx, batch_)) {
      throw std::runtime_error("Failed to eval");
    }

    n_past_ += size;
  }

  size_t n_past_;
  owned<llama_model> model_;
  owned<llama_context> ctx_;

  llama_batch batch_;
};

static int g_llama_cpp_log_level = 0;
static void llama_log_callback(ggml_log_level level, const char * text, void * user_data) {
  (void)user_data;
  if (level < g_llama_cpp_log_level) {
    fputs(text, stderr);
    fflush(stderr);
  }
}

struct BackendInitializer {
  BackendInitializer() {
    if (const char* level = std::getenv("LLAMA_CPP_LOG_LEVEL")) {
      g_llama_cpp_log_level = std::stoi(level);
    }
    llama_log_set(llama_log_callback, nullptr);
    llama_backend_init(false);
  }

  ~BackendInitializer() {
    llama_backend_free();
  }
};
} // namespace

std::unique_ptr<TextInferenceEngine> create_engine(rust::Str model_path) {
  static BackendInitializer initializer;

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 1;
  llama_model* model = llama_load_model_from_file(std::string(model_path).c_str(), model_params);

  if (!model) {
    return nullptr;
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = N_CTX;
  ctx_params.n_batch = N_BATCH;
  llama_context* ctx = llama_new_context_with_model(model, ctx_params);

  return std::make_unique<TextInferenceEngineImpl>(
      owned<llama_model>(model, llama_free_model),
      owned<llama_context>(ctx, llama_free)
  );
}

}  // namespace tabby
