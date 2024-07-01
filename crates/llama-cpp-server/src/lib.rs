mod supervisor;

use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::Deserialize;
use supervisor::LlamaCppSupervisor;
use tabby_common::{
    api::chat::Message,
    config::{HttpModelConfigBuilder, LocalModelConfig, ModelConfig},
    registry::{parse_model_id, ModelRegistry, GGML_MODEL_RELATIVE_PATH},
};
use tabby_inference::{
    ChatCompletionOptions, ChatCompletionStream, CompletionOptions, CompletionStream, Embedding,
};

fn api_endpoint(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

struct EmbeddingServer {
    #[allow(unused)]
    server: LlamaCppSupervisor,
    embedding: Arc<dyn Embedding>,
}

impl EmbeddingServer {
    async fn new(
        num_gpu_layers: u16,
        model_path: &str,
        parallelism: u8,
        enable_fast_attention: bool,
    ) -> EmbeddingServer {
        let server = LlamaCppSupervisor::new(
            "embedding",
            num_gpu_layers,
            true,
            model_path,
            parallelism,
            None,
            enable_fast_attention,
        );
        server.start().await;

        let config = HttpModelConfigBuilder::default()
            .api_endpoint(api_endpoint(server.port()))
            .kind("llama.cpp/embedding".to_string())
            .build()
            .expect("Failed to create HttpModelConfig");

        Self {
            server,
            embedding: http_api_bindings::create_embedding(&config).await,
        }
    }
}

#[async_trait]
impl Embedding for EmbeddingServer {
    async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
        self.embedding.embed(prompt).await
    }
}

struct CompletionServer {
    #[allow(unused)]
    server: LlamaCppSupervisor,
    completion: Arc<dyn CompletionStream>,
}

impl CompletionServer {
    async fn new(
        num_gpu_layers: u16,
        model_path: &str,
        parallelism: u8,
        enable_fast_attention: bool,
    ) -> Self {
        let server = LlamaCppSupervisor::new(
            "completion",
            num_gpu_layers,
            false,
            model_path,
            parallelism,
            None,
            enable_fast_attention,
        );
        server.start().await;
        let config = HttpModelConfigBuilder::default()
            .api_endpoint(api_endpoint(server.port()))
            .kind("llama.cpp/completion".to_string())
            .build()
            .expect("Failed to create HttpModelConfig");
        let completion = http_api_bindings::create(&config).await;
        Self { server, completion }
    }
}

#[async_trait]
impl CompletionStream for CompletionServer {
    async fn generate(&self, prompt: &str, options: CompletionOptions) -> BoxStream<String> {
        self.completion.generate(prompt, options).await
    }
}

struct ChatCompletionServer {
    #[allow(unused)]
    server: LlamaCppSupervisor,
    chat_completion: Arc<dyn ChatCompletionStream>,
}

impl ChatCompletionServer {
    async fn new(
        num_gpu_layers: u16,
        model_path: &str,
        parallelism: u8,
        chat_template: String,
        enable_fast_attention: bool,
    ) -> Self {
        let server = LlamaCppSupervisor::new(
            "chat",
            num_gpu_layers,
            false,
            model_path,
            parallelism,
            Some(chat_template),
            enable_fast_attention,
        );
        server.start().await;
        let config = HttpModelConfigBuilder::default()
            .api_endpoint(api_endpoint(server.port()))
            .kind("openai/chat".to_string())
            .build()
            .expect("Failed to create HttpModelConfig");
        let chat_completion = http_api_bindings::create_chat(&config).await;
        Self {
            server,
            chat_completion,
        }
    }
}

#[async_trait]
impl ChatCompletionStream for ChatCompletionServer {
    async fn chat_completion(
        &self,
        messages: &[Message],
        options: ChatCompletionOptions,
    ) -> Result<BoxStream<String>> {
        self.chat_completion
            .chat_completion(messages, options)
            .await
    }
}

pub async fn create_chat_completion(config: &LocalModelConfig) -> Arc<dyn ChatCompletionStream> {
    let model_path = resolve_model_path(&config.model_id).await;
    let info = resolve_prompt_info(&config.model_id).await;
    let chat_template = info
        .chat_template
        .unwrap_or_else(|| panic!("Chat model requires specifying prompt template"));
    Arc::new(
        ChatCompletionServer::new(
            config.num_gpu_layers,
            &model_path,
            config.parallelism,
            chat_template,
            config.enable_fast_attention,
        )
        .await,
    )
}

pub async fn create_completion(
    config: &LocalModelConfig,
) -> (Arc<dyn CompletionStream>, PromptInfo) {
    let model_path = resolve_model_path(&config.model_id).await;
    let prompt_info = resolve_prompt_info(&config.model_id).await;
    let stream = Arc::new(
        CompletionServer::new(
            config.num_gpu_layers,
            &model_path,
            config.parallelism,
            config.enable_fast_attention,
        )
        .await,
    );

    (stream, prompt_info)
}

pub async fn create_embedding(config: &ModelConfig) -> Arc<dyn Embedding> {
    match config {
        ModelConfig::Http(http) => http_api_bindings::create_embedding(http).await,
        ModelConfig::Local(llama) => {
            let model_path = resolve_model_path(&llama.model_id).await;
            Arc::new(
                EmbeddingServer::new(
                    llama.num_gpu_layers,
                    &model_path,
                    llama.parallelism,
                    llama.enable_fast_attention,
                )
                .await,
            )
        }
    }
}

async fn resolve_model_path(model_id: &str) -> String {
    let path = PathBuf::from(model_id);
    let path = if path.exists() {
        path.join(GGML_MODEL_RELATIVE_PATH)
    } else {
        let (registry, name) = parse_model_id(model_id);
        let registry = ModelRegistry::new(registry).await;
        registry.get_model_path(name)
    };
    path.display().to_string()
}

#[derive(Deserialize)]
pub struct PromptInfo {
    pub prompt_template: Option<String>,
    pub chat_template: Option<String>,
}

impl PromptInfo {
    fn read(filepath: PathBuf) -> PromptInfo {
        serdeconv::from_json_file(&filepath)
            .unwrap_or_else(|_| panic!("Invalid metadata file: {}", filepath.display()))
    }
}

async fn resolve_prompt_info(model_id: &str) -> PromptInfo {
    let path = PathBuf::from(model_id);
    if path.exists() {
        PromptInfo::read(path.join("tabby.json"))
    } else {
        let (registry, name) = parse_model_id(model_id);
        let registry = ModelRegistry::new(registry).await;
        let model_info = registry.get_model_info(name);
        PromptInfo {
            prompt_template: model_info.prompt_template.to_owned(),
            chat_template: model_info.chat_template.to_owned(),
        }
    }
}
