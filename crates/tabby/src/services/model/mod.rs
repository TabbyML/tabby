use std::{fs, sync::Arc};

pub use llama_cpp_server::PromptInfo;
use tabby_common::config::ModelConfig;
use tabby_download::download_model;
use tabby_inference::{ChatCompletionStream, CodeGeneration, CompletionStream, Embedding};
use tracing::info;

pub async fn load_chat_completion(chat: &ModelConfig) -> Arc<dyn ChatCompletionStream> {
    match chat {
        ModelConfig::Http(http) => http_api_bindings::create_chat(http).await,
        ModelConfig::Local(llama) => llama_cpp_server::create_chat_completion(llama).await,
    }
}

pub async fn load_embedding(config: &ModelConfig) -> Arc<dyn Embedding> {
    llama_cpp_server::create_embedding(config).await
}

pub async fn load_code_generation(model: &ModelConfig) -> (Arc<CodeGeneration>, PromptInfo) {
    let (engine, prompt_info) = load_completion(model).await;
    (Arc::new(CodeGeneration::new(engine)), prompt_info)
}

async fn load_completion(model: &ModelConfig) -> (Arc<dyn CompletionStream>, PromptInfo) {
    match model {
        ModelConfig::Http(http) => {
            let engine = http_api_bindings::create(http).await;
            let (prompt_template, chat_template) = http_api_bindings::build_completion_prompt(http);
            (
                engine,
                PromptInfo {
                    prompt_template,
                    chat_template,
                },
            )
        }
        ModelConfig::Local(llama) => llama_cpp_server::create_completion(llama).await,
    }
}

pub async fn download_model_if_needed(model: &str) {
    if fs::metadata(model).is_ok() {
        info!("Loading model from local path {}", model);
    } else {
        download_model(model, true).await;
    }
}
