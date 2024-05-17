mod supervisor;

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;
use supervisor::LlamaCppSupervisor;
use tabby_common::config::HttpModelConfigBuilder;
use tabby_inference::{CompletionOptions, CompletionStream, Embedding};

fn api_endpoint(port: u16) -> String {
    format!("http://localhost:{port}")
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
            embedding: http_api_bindings::create_embedding(&config),
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
        let config = HttpModelConfigBuilder::default()
            .api_endpoint(api_endpoint(server.port()))
            .kind("llama.cpp/completion".to_string())
            .build()
            .expect("Failed to create HttpModelConfig");
        let completion = http_api_bindings::create(&config);
        Self { server, completion }
    }
}

#[async_trait]
impl CompletionStream for CompletionServer {
    async fn generate(&self, prompt: &str, options: CompletionOptions) -> BoxStream<String> {
        self.completion.generate(prompt, options).await
    }
}

pub async fn create_embedding(
    num_gpu_layers: u16,
    model_path: &str,
    parallelism: u8,
) -> Arc<dyn Embedding> {
    Arc::new(EmbeddingServer::new(num_gpu_layers, model_path, parallelism).await)
}

pub async fn create_completion(
    num_gpu_layers: u16,
    model_path: &str,
    parallelism: u8,
) -> Arc<dyn CompletionStream> {
    Arc::new(CompletionServer::new(num_gpu_layers, model_path, parallelism).await)
}
