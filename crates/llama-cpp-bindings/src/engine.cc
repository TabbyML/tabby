#include "engine.h"

#include <functional>

#include <ggml.h>
#include <llama.h>
#include <common/common.h>

namespace tabby {
LlamaEngine::~LlamaEngine() {}

namespace {
template<class T>
using owned = std::unique_ptr<T, std::function<void(T*)>>;

class LlamaEngineImpl : public LlamaEngine {
 public:
  LlamaEngineImpl(owned<llama_model> model, owned<llama_context> ctx) :
    model_(std::move(model)),
    ctx_(std::move(ctx)) {
  }

  rust::Vec<uint32_t> inference(
      const rust::Str prompt,
      size_t max_decoding_length,
      float sampling_temperature
  ) const override {
    auto* ctx = ctx_.get();
    std::vector<llama_token> tokens_list = llama_tokenize(ctx, std::string(prompt), true);

    rust::Vec<uint32_t> ret;
    for (size_t n_remain = max_decoding_length; n_remain > 0; --n_remain) {
      if (llama_eval(
            ctx,
            tokens_list.data(),
            tokens_list.size(),
            llama_get_kv_cache_token_count(ctx),
            /* n_threads = */ 1)) {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        return {};
      }

      tokens_list.clear();

      auto logits = llama_get_logits(ctx);
      auto n_vocab = llama_n_vocab(ctx);

      std::vector<llama_token_data> candidates;
      candidates.reserve(n_vocab);
      for (llama_token token_id = 0; token_id < n_vocab; ++token_id) {
        candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
      }

      llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

      llama_token new_token_id = llama_sample_token_greedy(ctx , &candidates_p);

      if (new_token_id == llama_token_eos(ctx)) {
        break;
      }

      fprintf(stderr, "Next Token: %d\n", new_token_id);
      tokens_list.push_back(new_token_id);
      ret.push_back(new_token_id);
    }

    return ret;
  }

 private:
  owned<llama_model> model_;
  owned<llama_context> ctx_;
};

struct BackendInitializer {
  BackendInitializer() {
    llama_backend_init(false);
  }

  ~BackendInitializer() {
    llama_backend_free();
  }
};
} // namespace

std::shared_ptr<LlamaEngine> create_engine(rust::Str model_path) {
  static BackendInitializer initializer;

  llama_context_params ctx_params = llama_context_default_params();
  llama_model* model = llama_load_model_from_file(std::string(model_path).c_str(), ctx_params);

  if (!model) {
    fprintf(stderr , "%s: error: unable to load model\n" , __func__);
    return nullptr;
  }

  llama_context* ctx = llama_new_context_with_model(model, ctx_params);

  return std::make_unique<LlamaEngineImpl>(
      owned<llama_model>(model, llama_free_model),
      owned<llama_context>(ctx, llama_free)
  );
}

}  // namespace tabby
