#include "engine.h"

#include <functional>

#include <ggml.h>
#include <llama.h>

namespace tabby {
LlamaEngine::~LlamaEngine() {}

namespace {
template<class T>
using owned = std::unique_ptr<T, std::function<void(T*)>>;

class LlamaEngineImpl : public LlamaEngine {
 public:
  LlamaEngineImpl(owned<llama_model> model, owned<llama_context> context) :
    model_(std::move(model)),
    context_(std::move(context)) {
  }

  rust::Vec<uint32_t> inference(
      rust::Slice<const rust::String> tokens,
      size_t max_decoding_length,
      float sampling_temperature
  ) const override {
    // FIXME(meng): implement inference loop.
  }

 private:
  owned<llama_model> model_;
  owned<llama_context> context_;
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

  llama_context* context = llama_new_context_with_model(model, ctx_params);

  return std::make_unique<LlamaEngineImpl>(
      owned<llama_model>(model, llama_free_model),
      owned<llama_context>(context, llama_free)
  );
}

}  // namespace tabby
