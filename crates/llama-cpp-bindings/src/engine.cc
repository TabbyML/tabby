#include <functional>
#include <vector>
#include <deque>
#include <unordered_set>
#include <mutex>

#include <ggml.h>
#include <llama.h>

#include "llama-cpp-bindings/include/engine.h"
#include "llama-cpp-bindings/src/lib.rs.h"

namespace llama {
TextInferenceEngine::~TextInferenceEngine() {}

namespace {
constexpr size_t N_BATCH = 512;  // # per batch inference.
constexpr size_t N_CTX = 4096;   // # max kv history.
struct Request {
  Request(size_t request_id, std::vector<llama_token> input_token_ids) :
    id(request_id),
    tokens(input_token_ids.begin(), input_token_ids.end()) {
    }

  uint32_t id = -1;
  llama_seq_id seq_id = -1;

  std::vector<llama_token> tokens;
  size_t i_batch = -1;
  size_t n_past = 0;

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
  TextInferenceEngineImpl(owned<llama_model> model, owned<llama_context> ctx, uint8_t parallelism) :
    model_(std::move(model)),
    ctx_(std::move(ctx)),
    parallelism_(parallelism) {
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
    std::lock_guard<std::mutex> guard(g_mutex_);

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
        throw std::runtime_error(string_format("llama_decode failed with code: %d", ret));
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

        bool incomplete = false;
        for (size_t i = 1; i < 5 && i <= request.generated_text.size(); ++i)
        {
            const char c = request.generated_text[request.generated_text.size() - i];
            if ((c & 0xC0) == 0x80)
            {
                // continuation byte: 10xxxxxx
                continue;
            }
            else if ((c & 0xE0) == 0xC0)
            {
                // 2-byte character: 110xxxxx ...
                incomplete = i < 2;
            }
            else if ((c & 0xF0) == 0xE0)
            {
                // 3-byte character: 1110xxxx ...
                incomplete = i < 3;
            }
            else if ((c & 0xF8) == 0xF0)
            {
                // 4-byte character: 11110xxx ...
                incomplete = i < 4;
            }
            // else 1-byte character or invalid byte
            break;
        }

        if (!incomplete) {
          rust::String generated_text;
          try {
            generated_text = is_eos ? "" : request.generated_text;
          } catch (const std::invalid_argument& e) {
            fprintf(stderr, "%s:%d [%s] - ignoring non utf-8/utf-16 output\n", __FILE__, __LINE__, __func__);
          }

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

  uint32_t parallelism_;

  // llama.cpp is not thread safe
  // FIXME(meng): remove the mutex once https://github.com/ggerganov/llama.cpp/issues/3960 is fixed
  // and integrated to tabby's fork.
  static std::mutex g_mutex_;
};

std::mutex TextInferenceEngineImpl::g_mutex_;

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

std::unique_ptr<TextInferenceEngine> create_engine(bool use_gpu, rust::Str model_path, uint8_t parallelism) {
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
      parallelism
  );
}

}  // namespace tabby
