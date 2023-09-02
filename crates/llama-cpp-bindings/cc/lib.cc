#include <functional>

#include <ggml.h>
#include <llama.h>

class LlamaEngine {
 public:
  struct Options {
  };

  static std::unique_ptr<LlamaEngine> Create(const Options& options);

 private:
  template<class T>
  using owned = std::unique_ptr<T, std::function<void(T*)>>;

  owned<llama_model> model_;
  owned<llama_context> context_;
};


namespace {
struct BackendInitializer {
  BackendInitializer() {
    llama_backend_init(false);
  }

  ~BackendInitializer() {
    llama_backend_free();
  }
};
} // namespace

std::unique_ptr<LlamaEngine> LlamaEngine::Create(const Options& options) {
  static BackendInitializer initializer;

  llama_context_params ctx_params = llama_context_default_params();
  llama_model* model = llama_load_model_from_file("/abc", ctx_params);

  if (!model) {
    fprintf(stderr , "%s: error: unable to load model\n" , __func__);
    return nullptr;
  }

  llama_context* context = llama_new_context_with_model(model, ctx_params);

  auto engine = std::make_unique<LlamaEngine>();
  engine->model_ = owned<llama_model>(model, llama_free_model);
  engine->context_ = owned<llama_context>(context, llama_free);
  return engine;
}
