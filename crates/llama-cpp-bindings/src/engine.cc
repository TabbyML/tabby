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
constexpr size_t N_BATCH = 512;  // # per batch inference.
constexpr size_t N_CTX = 4096;   // # max kv history.

constexpr int DRAFT_N_GRAM_SIZE = 3;
constexpr int DRAFT_N_PRED_TOKENS = 10;
 
struct Request {
  Request(size_t request_id, std::vector<llama_token> input_token_ids) :
    id(request_id),
    tokens(input_token_ids.begin(), input_token_ids.end()) {
  }

  uint32_t id = -1;
  llama_seq_id seq_id = -1;

  std::vector<llama_token> tokens;
  size_t i_batch = -1;
  size_t n_draft = 0;

  int32_t multibyte_pending = 0;
  std::string generated_text;


  void draft_tokens(int n_draft_quota) {
    if (n_draft_quota < DRAFT_N_PRED_TOKENS) {
      n_draft = 0;
      return;
    }

    auto draft = find_candidate_pred_tokens(DRAFT_N_GRAM_SIZE, DRAFT_N_PRED_TOKENS);
    n_draft = draft.size();
    tokens.insert(tokens.end(), draft.begin(), draft.end());
  }

  size_t n_past() {
    return past_tokens.size();
  }

  void step(llama_token next_token, size_t n_dropped) {
    past_tokens.insert(past_tokens.end(), tokens.begin(), tokens.end() - n_dropped);

    n_draft = 0;
    tokens.clear();
    tokens.push_back(next_token);
  }

 private:
  std::vector<llama_token> find_candidate_pred_tokens(size_t ngram_size, size_t n_pred_tokens) {
    auto ngram = build_ngram(ngram_size);
    if (ngram.size() < ngram_size) return {};

    const auto end = past_tokens.end() - ngram_size - n_pred_tokens;
    const auto matched = std::search(past_tokens.begin(), end, ngram.begin(), ngram.end());
    if (matched == end) return {};

    const auto begin = matched + ngram_size;
    return std::vector<llama_token>(begin, begin + n_pred_tokens);
  }

  std::vector<llama_token> build_ngram(size_t ngram_size) {
    GGML_ASSERT(n_draft == 0);
    std::deque<llama_token> ret;
    for (int i = tokens.size() - 1; i >= 0; --i) {
      if (ret.size() == ngram_size) break;
      ret.push_front(tokens[i]);
    }

    for (int i = past_tokens.size() - 1; i >= 0; --i) {
      if (ret.size() == ngram_size) break;
      ret.push_front(past_tokens[i]);
    }

    return std::vector<llama_token>(ret.begin(), ret.end());
  }

  std::vector<llama_token> past_tokens;
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

template<typename ... Args>
std::string string_format(const std::string& format, Args ... args)
{
	int size_s = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
	if (size_s <= 0) { throw std::runtime_error("Error during formatting."); }
	auto size = static_cast<size_t>(size_s);
	std::unique_ptr<char[]> buf(new char[size]);
	std::snprintf(buf.get(), size, format.c_str(), args ...);
	return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

template<class T>
using owned = std::unique_ptr<T, std::function<void(T*)>>;

class TextInferenceEngineImpl : public TextInferenceEngine {
 public:
  TextInferenceEngineImpl(owned<llama_model> model, owned<llama_context> ctx, uint8_t parallelism, bool enable_prompt_lookup) :
    model_(std::move(model)),
    ctx_(std::move(ctx)),
    parallelism_(parallelism),
    enable_prompt_lookup_(enable_prompt_lookup) {
      batch_ = llama_batch_init(N_CTX * parallelism, 0, 1);
      // warm up
      {
        batch_.n_tokens = 16;
        for (int i = 0; i < batch_.n_tokens; ++i) {
          batch_.token[i] = 0;
          batch_.pos[i] = i;
          batch_.n_seq_id[i] = 1;
          batch_.seq_id[i][0] = 0;
          batch_.logits[i] = false;
        }

        if (llama_decode(ctx_.get(), batch_)) {
          fprintf(stderr, "%s: warmup failed\n", __func__);
        }

        llama_kv_cache_clear(ctx_.get());
      }
  }

  ~TextInferenceEngineImpl() {
    llama_batch_free(batch_);
  }

  virtual void add_request(uint32_t request_id, rust::Str text, size_t max_input_length) override {
    auto tokens = llama_tokenize(llama_get_model(ctx_.get()), text, false, true);
    if (tokens.size() > max_input_length) {
      int start = tokens.size() - max_input_length;
      tokens = std::vector<llama_token>(tokens.begin() + start, tokens.end());
    }
    pending_requests_.push_back(Request(request_id, tokens));
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
    while (pending_requests_.size() > 0 && requests_.size() < parallelism_) {
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

      // Ensure the draft logits always fall into the same batch.
      if (enable_prompt_lookup_) {
        const int n_draft_quota = N_BATCH - (n_tokens + request.tokens.size()) % N_BATCH;
        request.draft_tokens(n_draft_quota);
      }

      for (size_t i = 0; i < request.tokens.size(); ++i) {
        batch_.token[n_tokens + i] = request.tokens[i];
        batch_.pos[n_tokens + i] = request.n_past() + i;
        batch_.n_seq_id[n_tokens + i] = 1;
        batch_.seq_id[n_tokens + i][0] = request.id;
        batch_.logits[n_tokens + i] = false;
      }
      batch_.n_tokens += request.tokens.size();

      for (int k = batch_.n_tokens - request.n_draft - 1; k <= batch_.n_tokens - 1; ++ k) {
        batch_.logits[k] = true;
      }
      request.i_batch = batch_.n_tokens;
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
        throw std::runtime_error(string_format("llama_decode failed with code: %d", ret));
      }

      const auto eos_id = llama_token_eos(llama_get_model(ctx));
      for (auto& request : requests_) {
        int32_t i_batch = request.i_batch - i - 1;
        if ((i_batch < 0) || (i_batch >= n_tokens)) {
          continue;
        }

        for (int k = -request.n_draft; k < 1; ++k) {
          auto logits = llama_get_logits_ith(ctx, i_batch + k);
          llama_token next_token = std::distance(logits, std::max_element(logits, logits + n_vocab));

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
            rust::String generated_text;
            try {
              generated_text = is_eos ? "" : request.generated_text;
            } catch (const std::invalid_argument& e) {
              fprintf(stderr, "%s:%d [%s] - ignoring non utf-8/utf-16 output\n", __FILE__, __LINE__, __func__);
            }

            result.push_back({request.id, generated_text});
            request.generated_text.clear();
          }

          if (is_eos) {
            break;
          }

          if ((k == 0) || ((k < 0 && next_token != request.tokens[request.tokens.size() + k]))) {
            request.step(next_token, -k);
            llama_kv_cache_seq_rm(ctx_.get(), request.id, request.n_past(), -1);
            break;
          }
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

  uint32_t parallelism_;
  bool enable_prompt_lookup_;
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

std::unique_ptr<TextInferenceEngine> create_engine(
  bool use_gpu,
  rust::Str model_path,
  uint8_t parallelism,
  bool enable_prompt_lookup
) {
  static BackendInitializer initializer;

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = use_gpu ? 9999 : 0;
  llama_model* model = llama_load_model_from_file(std::string(model_path).c_str(), model_params);

  if (!model) {
    return nullptr;
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = N_CTX * parallelism;
  ctx_params.n_batch = N_BATCH;
  if (const char* n_thread_str = std::getenv("LLAMA_CPP_N_THREADS")) {
    int n_threads = std::stoi(n_thread_str);
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
  }
  llama_context* ctx = llama_new_context_with_model(model, ctx_params);
  return std::make_unique<TextInferenceEngineImpl>(
      owned<llama_model>(model, llama_free_model),
      owned<llama_context>(ctx, llama_free),
      parallelism,
      enable_prompt_lookup
  );
}

}  // namespace tabby
