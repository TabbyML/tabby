#include "engine.h"

#include <functional>
#include <vector>

#include <ggml.h>
#include <llama.h>

namespace llama {
TextInferenceEngine::~TextInferenceEngine() {}

namespace {
static size_t N_BATCH = 512;

template<class T>
using owned = std::unique_ptr<T, std::function<void(T*)>>;

std::vector<llama_token> tokenize(struct llama_context * ctx, const std::string & text, size_t max_input_length, bool add_bos) {
    const struct llama_model* model = llama_get_model(ctx);
    // upper limit for the number of tokens
    int n_tokens = max_input_length;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_bos);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_bos);
        GGML_ASSERT(check == -n_tokens);

        int start = check - max_input_length;
        GGML_ASSERT(start >= 0);
        result = std::vector<llama_token>(result.begin() + start, result.end());
        if (add_bos) {
          result[0] = llama_token_bos(ctx);
        }
    } else {
        result.resize(n_tokens);
    }
    return result;
}

class TextInferenceEngineImpl : public TextInferenceEngine {
 public:
  TextInferenceEngineImpl(owned<llama_model> model, owned<llama_context> ctx) :
    model_(std::move(model)),
    ctx_(std::move(ctx)) {
  }

  std::unique_ptr<std::vector<uint32_t>> start(const rust::Str prompt, size_t max_input_length) const override {
    auto* ctx = ctx_.get();
    llama_reset_timings(ctx);
    std::vector<llama_token> tokens_list = tokenize(ctx, std::string(prompt), max_input_length, /* add_bos = */ false);

    for (size_t i = 0; i < tokens_list.size(); i += N_BATCH) {
      const size_t size = std::min(N_BATCH, tokens_list.size() - i);
      eval(tokens_list.data() + i, size, /* reset = */ i == 0);
    }

    return std::make_unique<std::vector<uint32_t>>(tokens_list.begin(), tokens_list.end());
  }

  uint32_t step() const override {
    const llama_token id = sample();
    eval(const_cast<llama_token*>(&id), 1, /* reset = */ false);
    return id;
  }

  void end() const override {
    llama_print_timings(ctx_.get());
  }

  uint32_t eos_token() const override {
    return llama_token_eos(ctx_.get());
  }

 private:
  uint32_t sample() const {
    auto* ctx = ctx_.get();

    auto logits = llama_get_logits(ctx);
    auto n_vocab = llama_n_vocab(llama_get_model(ctx));

    // Greedy sampling (always select the highest logit).
    return std::distance(logits, std::max_element(logits, logits + n_vocab));
  }

  bool eval(llama_token* data, size_t size, bool reset) const {
    auto* ctx = ctx_.get();
    if (llama_eval(
          ctx,
          data,
          size,
          reset ? 0 : llama_get_kv_cache_token_count(ctx))) {
      fprintf(stderr, "%s : failed to eval\n", __func__);
      return false;
    }

    return true;
  }

  owned<llama_model> model_;
  owned<llama_context> ctx_;
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

std::shared_ptr<TextInferenceEngine> create_engine(rust::Str model_path) {
  static BackendInitializer initializer;

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 1;
  llama_model* model = llama_load_model_from_file(std::string(model_path).c_str(), model_params);

  if (!model) {
    fprintf(stderr , "%s: error: unable to load model\n" , __func__);
    return nullptr;
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = N_BATCH;
  llama_context* ctx = llama_new_context_with_model(model, ctx_params);

  return std::make_shared<TextInferenceEngineImpl>(
      owned<llama_model>(model, llama_free_model),
      owned<llama_context>(ctx, llama_free)
  );
}

}  // namespace tabby
