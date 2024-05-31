mod supervisor;

use std::{fs, path::PathBuf, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;
use supervisor::LlamaCppSupervisor;
use tabby_common::{
    config::{HttpModelConfigBuilder, ModelConfig},
    registry::{parse_model_id, ModelRegistry, GGML_MODEL_RELATIVE_PATH},
};
use tabby_inference::{CompletionOptions, CompletionStream, Embedding};

fn api_endpoint(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

struct EmbeddingServer {
    #[allow(unused)]
    server: LlamaCppSupervisor,
    embedding: Arc<dyn Embedding>,
}

impl EmbeddingServer {
    async fn new(num_gpu_layers: u16, model_path: &str, parallelism: u8) -> EmbeddingServer {
        let server = LlamaCppSupervisor::new(num_gpu_layers, true, model_path, parallelism);
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
    async fn new(num_gpu_layers: u16, model_path: &str, parallelism: u8) -> Self {
        let server = LlamaCppSupervisor::new(num_gpu_layers, false, model_path, parallelism);
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

pub async fn create_completion(
    num_gpu_layers: u16,
    model_path: &str,
    parallelism: u8,
) -> Arc<dyn CompletionStream> {
    Arc::new(CompletionServer::new(num_gpu_layers, model_path, parallelism).await)
}

pub async fn create_embedding(config: &ModelConfig) -> Arc<dyn Embedding> {
    match config {
        ModelConfig::Http(http) => http_api_bindings::create_embedding(http).await,
        ModelConfig::Local(llama) => {
            if fs::metadata(&llama.model_id).is_ok() {
                let path = PathBuf::from(&llama.model_id);
                let model_path = path.join(GGML_MODEL_RELATIVE_PATH);
                Arc::new(
                    EmbeddingServer::new(
                        llama.num_gpu_layers,
                        model_path.display().to_string().as_str(),
                        llama.parallelism,
                    )
                    .await,
                )
            } else {
                let (registry, name) = parse_model_id(&llama.model_id);
                let registry = ModelRegistry::new(registry).await;
                let model_path = registry.get_model_path(name).display().to_string();
                Arc::new(
                    EmbeddingServer::new(llama.num_gpu_layers, &model_path, llama.parallelism)
                        .await,
                )
            }
        }
    }
}
