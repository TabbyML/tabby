use std::{fs, sync::Arc};

pub use llama_cpp_server::PromptInfo;
use tabby_common::config::ModelConfig;
use tabby_download::download_model;
use tabby_inference::{ChatCompletionStream, CodeGeneration, CompletionStream, Embedding};
use tracing::info;

pub async fn load_embedding(config: &ModelConfig) -> Arc<dyn Embedding> {
    llama_cpp_server::create_embedding(config).await
}

pub async fn load_code_generation_and_chat(
    completion_model: Option<ModelConfig>,
    chat_model: Option<ModelConfig>,
) -> (
    Option<Arc<CodeGeneration>>,
    Option<PromptInfo>,
    Option<Arc<dyn ChatCompletionStream>>,
) {
    let (engine, prompt_info, chat) =
        load_completion_and_chat(completion_model.clone(), chat_model).await;
    let code = engine.map(|engine| Arc::new(CodeGeneration::new(engine, completion_model)));
    (code, prompt_info, chat)
}

async fn load_completion_and_chat(
    completion_model: Option<ModelConfig>,
    chat_model: Option<ModelConfig>,
) -> (
    Option<Arc<dyn CompletionStream>>,
    Option<PromptInfo>,
    Option<Arc<dyn ChatCompletionStream>>,
) {
    if let (Some(ModelConfig::Local(completion)), Some(ModelConfig::Local(chat))) =
        (&completion_model, &chat_model)
    {
        let (completion, prompt, chat) =
            llama_cpp_server::create_completion_and_chat(completion, chat).await;
        return (Some(completion), Some(prompt), Some(chat));
    }

    let (completion, prompt) = if let Some(completion_model) = completion_model {
        match completion_model {
            ModelConfig::Http(http) => {
                let engine = http_api_bindings::create(&http).await;
                let (prompt_template, chat_template) =
                    http_api_bindings::build_completion_prompt(&http);
                (
                    Some(engine),
                    Some(PromptInfo {
                        prompt_template,
                        chat_template,
                    }),
                )
            }
            ModelConfig::Local(llama) => {
                let (stream, prompt) = llama_cpp_server::create_completion(&llama).await;
                (Some(stream), Some(prompt))
            }
        }
    } else {
        (None, None)
    };

    let chat = if let Some(chat_model) = chat_model {
        match chat_model {
            ModelConfig::Http(http) => Some(http_api_bindings::create_chat(&http).await),
            ModelConfig::Local(llama) => {
                Some(llama_cpp_server::create_chat_completion(&llama).await)
            }
        }
    } else {
        None
    };

    (completion, prompt, chat)
}

pub async fn download_model_if_needed(model: &str) {
    if fs::metadata(model).is_ok() {
        info!("Loading model from local path {}", model);
    } else {
        download_model(model, true).await;
    }
}
