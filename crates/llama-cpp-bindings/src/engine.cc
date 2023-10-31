#include "engine.h"

#include <functional>
#include <vector>
#include <deque>
#include <unordered_set>

#include <ggml.h>
#include <llama.h>

#include "llama-cpp-bindings/src/lib.rs.h"

namespace llama {
TextInferenceEngine::~TextInferenceEngine() {}

namespace {
int get_parallelism() {
  const char* parallelism = std::getenv("LLAMA_CPP_PARALLELISM");
  if (parallelism) {
    return std::stoi(parallelism);
  } else {
    return 4;
  }
}

static size_t N_CONCURRENT_REQUESTS = get_parallelism();

constexpr size_t N_BATCH = 512;  // # per batch inference.
constexpr size_t N_CTX = 4096;   // # max kv history.
 
struct Request {
  Request(size_t request_id, rust::Slice<const uint32_t> input_token_ids) :
    id(request_id),
    tokens(input_token_ids.begin(), input_token_ids.end()) {
    }

  uint32_t id = -1;
  llama_seq_id seq_id = -1;

  std::vector<llama_token> tokens;
  size_t i_batch = -1;
  size_t n_past = 0;

  int32_t multibyte_pending = 0;
  std::string generated_text;
};


std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size());
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size());
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

std::vector<llama_token> llama_tokenize(
    const struct llama_model * model,
           const rust::Str &   text,
                        bool   add_bos,
                        bool   special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + add_bos;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_bos, special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_bos, special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

template<class T>
using owned = std::unique_ptr<T, std::function<void(T*)>>;

class TextInferenceEngineImpl : public TextInferenceEngine {
 public:
  TextInferenceEngineImpl(owned<llama_model> model, owned<llama_context> ctx) :
    model_(std::move(model)),
    ctx_(std::move(ctx)) {
      batch_ = llama_batch_init(N_CTX * N_CONCURRENT_REQUESTS, 0, 1);
  }

  ~TextInferenceEngineImpl() {
    llama_batch_free(batch_);
  }

  rust::Vec<uint32_t> tokenize(rust::Str text) override {
    const auto tokens = llama_tokenize(llama_get_model(ctx_.get()), text, false, true);

    rust::Vec<uint32_t> ret;
    ret.reserve(tokens.size());
    std::copy(tokens.begin(), tokens.end(), std::back_inserter(ret));
    return ret;
  }

  void add_request(uint32_t request_id, rust::Slice<const uint32_t> input_token_ids) override {
    pending_requests_.push_back(Request(request_id, input_token_ids));
  }

  void stop_request(uint32_t request_id) override {
    stopped_requests_.insert(request_id);
  }

  rust::Vec<StepOutput> step() override {
    auto* ctx = ctx_.get();
    auto n_vocab = llama_n_vocab(llama_get_model(ctx));

    // Remove stopped requests.
    if (!stopped_requests_.empty()) {
      std::vector<Request> requests;
      for (auto& request : requests_) {
        if (stopped_requests_.count(request.id) > 0) {
          // Release KV cache.
          llama_kv_cache_seq_rm(ctx_.get(), request.id, -1, -1);
        } else {
          requests.emplace_back(request);
        }
      }

      requests_ = requests;
    }

    // Add pending requests.
    while (pending_requests_.size() > 0 && requests_.size() < N_CONCURRENT_REQUESTS) {
      Request request = std::move(pending_requests_.front());
      pending_requests_.pop_front();

      // Ignore stopped pending requests.
      if (stopped_requests_.count(request.id) > 0) {
        continue;
      }

      requests_.push_back(request);
    }

    // Clear stopped requests.
    stopped_requests_.clear();

    if (requests_.size() == 0) {
      llama_kv_cache_clear(ctx);
      return {};
    }

    // Clear the batch.
    batch_.n_tokens = 0;

    // Insert tokens from ongoing requests to batch.
    for (auto& request : requests_) {
      const size_t n_tokens = batch_.n_tokens;
      for (size_t i = 0; i < request.tokens.size(); ++i) {
        batch_.token[n_tokens + i] = request.tokens[i];
        batch_.pos[n_tokens + i] = request.n_past + i;
        batch_.n_seq_id[n_tokens + i] = 1;
        batch_.seq_id[n_tokens + i][0] = request.id;
        batch_.logits[n_tokens + i] = false;
      }
      batch_.n_tokens += request.tokens.size();

      batch_.logits[batch_.n_tokens - 1] = true;
      request.i_batch = batch_.n_tokens - 1;
    }

    rust::Vec<StepOutput> result;
    result.reserve(requests_.size());

    // Decode tokens in chunks
    for (size_t i = 0; i < static_cast<size_t>(batch_.n_tokens); i += N_BATCH) {
      const int32_t n_tokens = std::min(N_BATCH, batch_.n_tokens - i);
			llama_batch batch_view = {
				n_tokens,
				batch_.token    + i,
				nullptr,
				batch_.pos      + i,
				batch_.n_seq_id + i,
				batch_.seq_id   + i,
				batch_.logits   + i,
				0, 0, 0, // unused
			};

			const int ret = llama_decode(ctx, batch_view);
      if (ret != 0) {
        throw std::runtime_error("Failed to eval");
      }

      const auto eos_id = llama_token_eos(llama_get_model(ctx));
      for (auto& request : requests_) {
        if ((request.i_batch < i) || (request.i_batch >= (i + n_tokens))) {
          continue;
        }

        int32_t i_batch = request.i_batch - i;
        auto logits = llama_get_logits_ith(ctx, i_batch);
        auto next_token = std::distance(logits, std::max_element(logits, logits + n_vocab));

        request.n_past += request.tokens.size();

        request.tokens.clear();
        request.tokens.push_back(next_token);

        const auto token_str = llama_token_to_piece(ctx, next_token);
        request.generated_text += token_str;

        // FIXME: Hack for codellama to simplify tabby's implementation.
        const bool is_eos = next_token == eos_id || token_str == " <EOT>";

        if (request.multibyte_pending > 0) {
          request.multibyte_pending -= token_str.size();
        } else if (token_str.size() == 1) {
            const char c = token_str[0];
            // 2-byte characters: 110xxxxx 10xxxxxx
            if ((c & 0xE0) == 0xC0) {
                request.multibyte_pending = 1;
                // 3-byte characters: 1110xxxx 10xxxxxx 10xxxxxx
            }
            else if ((c & 0xF0) == 0xE0) {
                request.multibyte_pending = 2;
                // 4-byte characters: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
            } else if ((c & 0xF8) == 0xF0) {
                request.multibyte_pending = 3;
            }
            else {
                request.multibyte_pending = 0;
            }
        }

        if (request.multibyte_pending == 0) {
          rust::String generated_text = is_eos ? "" : request.generated_text;
          result.push_back({request.id, generated_text});

          request.generated_text.clear();
        }
      }
    }

    return result;
  }

 private:
  owned<llama_model> model_;
  owned<llama_context> ctx_;

  llama_batch batch_;

  std::vector<Request> requests_;
  std::deque<Request> pending_requests_;
  std::unordered_set<uint32_t> stopped_requests_;
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

std::unique_ptr<TextInferenceEngine> create_engine(bool use_gpu, rust::Str model_path) {
  static BackendInitializer initializer;

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = use_gpu ? 9999 : 0;
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
