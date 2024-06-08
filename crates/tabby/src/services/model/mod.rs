use std::{fs, path::PathBuf, sync::Arc};

use serde::Deserialize;
use tabby_common::{
    config::ModelConfig,
    registry::{parse_model_id, ModelRegistry, GGML_MODEL_RELATIVE_PATH},
};
use tabby_download::download_model;
use tabby_inference::{ChatCompletionStream, CodeGeneration, CompletionStream, Embedding};
use tracing::info;

use crate::fatal;

pub async fn load_chat_completion(chat: &ModelConfig) -> Arc<dyn ChatCompletionStream> {
    match chat {
        ModelConfig::Http(http) => http_api_bindings::create_chat(http).await,

        ModelConfig::Local(llama) => {
            if fs::metadata(&llama.model_id).is_ok() {
                let path = PathBuf::from(&llama.model_id);
                let model_path = path.join(GGML_MODEL_RELATIVE_PATH).display().to_string();
                let engine_info = PromptInfo::read(path.join("tabby.json"));
                llama_cpp_server::create_chat_completion(
                    llama.num_gpu_layers,
                    &model_path,
                    llama.parallelism,
                    engine_info.chat_template.unwrap_or_else(|| fatal!("Chat model requires specifying prompt template"))
                )
                .await
            } else {
                let (registry, name) = parse_model_id(&llama.model_id);
                let registry = ModelRegistry::new(registry).await;
                let model_path = registry.get_model_path(name).display().to_string();
                let model_info = registry.get_model_info(name);
                llama_cpp_server::create_chat_completion(
                    llama.num_gpu_layers,
                    &model_path,
                    llama.parallelism,
                    model_info.chat_template.clone().unwrap_or_else(|| fatal!("Chat model requires specifying prompt template"))
                )
                .await
            }
        }
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
        ModelConfig::Local(llama) => {
            if fs::metadata(&llama.model_id).is_ok() {
                let path = PathBuf::from(&llama.model_id);
                let model_path = path.join(GGML_MODEL_RELATIVE_PATH);
                let engine = create_ggml_engine(
                    llama.num_gpu_layers,
                    model_path.display().to_string().as_str(),
                    llama.parallelism,
                )
                .await;
                let engine_info = PromptInfo::read(path.join("tabby.json"));
                (engine, engine_info)
            } else {
                let (registry, name) = parse_model_id(&llama.model_id);
                let registry = ModelRegistry::new(registry).await;
                let model_path = registry.get_model_path(name).display().to_string();
                let model_info = registry.get_model_info(name);
                let engine =
                    create_ggml_engine(llama.num_gpu_layers, &model_path, llama.parallelism).await;
                (
                    engine,
                    PromptInfo {
                        prompt_template: model_info.prompt_template.clone(),
                        chat_template: model_info.chat_template.clone(),
                    },
                )
            }
        }
    }
}

#[derive(Deserialize)]
pub struct PromptInfo {
    pub prompt_template: Option<String>,
    pub chat_template: Option<String>,
}

impl PromptInfo {
    fn read(filepath: PathBuf) -> PromptInfo {
        serdeconv::from_json_file(&filepath)
            .unwrap_or_else(|_| fatal!("Invalid metadata file: {}", filepath.display()))
    }
}

async fn create_ggml_engine(
    num_gpu_layers: u16,
    model_path: &str,
    parallelism: u8,
) -> Arc<dyn CompletionStream> {
    llama_cpp_server::create_completion(num_gpu_layers, model_path, parallelism).await
}

pub async fn download_model_if_needed(model: &str) {
    if fs::metadata(model).is_ok() {
        info!("Loading model from local path {}", model);
    } else {
        download_model(model, true).await;
    }
}
