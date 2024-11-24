mod llama;
mod openai;
mod voyage;

use core::{panic, time};
use std::{
    sync::Arc,
    task::{Context, Poll},
};

use async_trait::async_trait;
use futures::future::BoxFuture;
use llama::LlamaCppEngine;
use tabby_common::config::HttpModelConfig;
use tabby_inference::Embedding;
use tokio::sync::Mutex;
use tower::{Service, ServiceBuilder, ServiceExt};
use tracing::debug;

use self::{openai::OpenAIEmbeddingEngine, voyage::VoyageEmbeddingEngine};

pub async fn create(config: &HttpModelConfig) -> Arc<dyn Embedding> {
    let rpm = if let Some(limit) = &config.request_limit {
        limit.request_per_minute
    } else {
        0
    };

    let embedding = match config.kind.as_str() {
        "llama.cpp/embedding" => {
            let engine = LlamaCppEngine::create(
                config
                    .api_endpoint
                    .as_deref()
                    .expect("api_endpoint is required"),
                config.api_key.clone(),
            );
            Arc::new(engine)
        }
        "ollama/embedding" => ollama_api_bindings::create_embedding(config).await,
        "openai/embedding" => {
            let engine = OpenAIEmbeddingEngine::create(
                config
                    .api_endpoint
                    .as_deref()
                    .expect("api_endpoint is required"),
                config.model_name.as_deref().unwrap_or_default(),
                config.api_key.as_deref(),
            );
            Arc::new(engine)
        }
        "voyage/embedding" => {
            let engine = VoyageEmbeddingEngine::create(
                config.api_endpoint.as_deref(),
                config
                    .model_name
                    .as_deref()
                    .expect("model_name must be set for voyage/embedding"),
                config
                    .api_key
                    .clone()
                    .expect("api_key must be set for voyage/embedding"),
            );
            Arc::new(engine)
        }
        unsupported_kind => panic!(
            "Unsupported kind for http embedding model: {}",
            unsupported_kind
        ),
    };

    if rpm > 0 {
        debug!(
            "Creating rate limited embedding with {} requests per minute",
            rpm,
        );
        Arc::new(
            RateLimitedEmbedding::new(embedding, rpm)
                .expect("Failed to create rate limited embedding"),
        )
    } else {
        embedding
    }
}

struct EmbeddingService {
    embedding: Arc<dyn Embedding>,
}

impl Service<String> for EmbeddingService {
    type Response = Vec<f32>;
    type Error = anyhow::Error;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, prompt: String) -> Self::Future {
        let embedding = self.embedding.clone();
        Box::pin(async move { embedding.embed(&prompt).await })
    }
}

pub struct RateLimitedEmbedding {
    embedding: Arc<Mutex<tower::util::BoxService<String, Vec<f32>, anyhow::Error>>>,
}

impl RateLimitedEmbedding {
    pub fn new(embedding: Arc<dyn Embedding>, rpm: u64) -> anyhow::Result<Self> {
        if rpm == 0 {
            anyhow::bail!(
                "Can not create rate limited embedding client with 0 requests per minute"
            );
        }

        let service = ServiceBuilder::new()
            .rate_limit(rpm, time::Duration::from_secs(60))
            .service(EmbeddingService { embedding })
            .boxed();

        Ok(Self {
            embedding: Arc::new(Mutex::new(service)),
        })
    }
}

#[async_trait]
impl Embedding for RateLimitedEmbedding {
    async fn embed(&self, prompt: &str) -> anyhow::Result<Vec<f32>> {
        let mut service = self.embedding.lock().await;
        let prompt_owned = prompt.to_string();
        let response = service.ready().await?.call(prompt_owned).await?;
        Ok(response)
    }
}
