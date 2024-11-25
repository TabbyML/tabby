use std::{
    sync::Arc,
    task::{Context, Poll},
    time,
};

use async_trait::async_trait;
use futures::future::BoxFuture;
use tabby_inference::Embedding;
use tokio::sync::Mutex;
use tower::{Service, ServiceBuilder, ServiceExt};

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
