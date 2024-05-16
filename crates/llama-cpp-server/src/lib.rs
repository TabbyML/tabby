mod supervisor;

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::json;
use supervisor::LlamaCppSupervisor;
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
    async fn new(use_gpu: bool, model_path: &str, parallelism: u8) -> EmbeddingServer {
        let server = LlamaCppSupervisor::new(use_gpu, true, model_path, parallelism);
        server.start().await;

        let model_spec: String = serde_json::to_string(&json!({
            "kind": "llama",
            "api_endpoint": api_endpoint(server.port()),
        }))
        .expect("Failed to serialize model spec");
        Self {
            server,
            embedding: http_api_bindings::create_embedding(&model_spec),
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
    async fn new(use_gpu: bool, model_path: &str, parallelism: u8) -> Self {
        let server = LlamaCppSupervisor::new(use_gpu, false, model_path, parallelism);
        let model_spec: String = serde_json::to_string(&json!({
            "kind": "llama",
            "api_endpoint": api_endpoint(server.port()),
        }))
        .expect("Failed to serialize model spec");
        let (completion, _, _) = http_api_bindings::create(&model_spec);
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
    use_gpu: bool,
    model_path: &str,
    parallelism: u8,
) -> Arc<dyn Embedding> {
    Arc::new(EmbeddingServer::new(use_gpu, model_path, parallelism).await)
}

pub async fn create_completion(
    use_gpu: bool,
    model_path: &str,
    parallelism: u8,
) -> Arc<dyn CompletionStream> {
    Arc::new(CompletionServer::new(use_gpu, model_path, parallelism).await)
}
