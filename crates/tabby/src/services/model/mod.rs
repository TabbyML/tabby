use std::{collections::hash_map, fs, sync::Arc};

use anyhow::{bail, Result};
pub use llama_cpp_server::PromptInfo;
use tabby_common::{
    config::ModelConfig,
    registry::{parse_model_id, ModelInfo, ModelRegistry},
};
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
    Option<Arc<dyn CompletionStream>>,
    Option<Arc<dyn ChatCompletionStream>>,
    Option<PromptInfo>,
) {
    let (engine, prompt_info, chat) =
        load_completion_and_chat(completion_model.clone(), chat_model).await;
    let code = engine
        .clone()
        .map(|engine| Arc::new(CodeGeneration::new(engine, completion_model)));
    (code, engine, chat, prompt_info)
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
pub struct Downloader {
    registries: hash_map::HashMap<String, ModelRegistry>,
}

impl Downloader {
    pub fn new() -> Self {
        Self {
            registries: hash_map::HashMap::new(),
        }
    }

    pub async fn get_model_registry_and_info(
        &mut self,
        model_id: &str,
    ) -> Result<(ModelRegistry, ModelInfo)> {
        let (registry_name, model_name) = parse_model_id(model_id)?;

        let registry = if let Some(registry) = self.registries.get(registry_name) {
            registry.clone()
        } else {
            let registry = ModelRegistry::new(&registry_name).await;
            self.registries
                .insert(registry_name.to_owned(), registry.clone());
            registry
        };

        let info = registry.get_model_info(model_name)?.clone();

        Ok((registry, info))
    }

    pub async fn download_model(
        &self,
        registry: &ModelRegistry,
        model_id: &str,
        prefer_local_file: bool,
    ) -> Result<()> {
        let (_, model_name) = parse_model_id(model_id)?;
        download_model(&registry, model_name, prefer_local_file).await
    }

    pub async fn download_completion(&mut self, model_id: &str) -> Result<()> {
        if fs::metadata(model_id).is_ok() {
            info!("Loading model from local path {}", model_id)
        }

        let (registry, info) = self.get_model_registry_and_info(model_id).await?;
        if info.prompt_template.is_none() {
            bail!("Model '{}' doesn't support completion", model_id);
        }

        self.download_model(&registry, model_id, true).await
    }

    pub async fn download_chat(&mut self, model_id: &str) -> Result<()> {
        if fs::metadata(model_id).is_ok() {
            info!("Loading model from local path {}", model_id)
        }

        let (registry, info) = self.get_model_registry_and_info(model_id).await?;
        if info.chat_template.is_none() {
            bail!("Model '{}' doesn't support chat", model_id);
        }

        self.download_model(&registry, model_id, true).await
    }

    pub async fn download_embedding(&mut self, model_id: &str) -> Result<()> {
        if fs::metadata(model_id).is_ok() {
            info!("Loading model from local path {}", model_id)
        }

        let (registry, _) = self.get_model_registry_and_info(model_id).await?;

        self.download_model(&registry, model_id, true).await
    }
}
