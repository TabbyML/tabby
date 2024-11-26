use async_trait::async_trait;
use ratelimit::Ratelimiter;
use tabby_inference::Embedding;

pub struct RateLimitedEmbedding {
    embedding: Box<dyn Embedding>,
    rate_limiter: Ratelimiter,
}

impl RateLimitedEmbedding {
    pub fn new(embedding: Box<dyn Embedding>, rate_limiter: Ratelimiter) -> Self {
        Self {
            embedding,
            rate_limiter,
        }
    }
}

#[async_trait]
impl Embedding for RateLimitedEmbedding {
    async fn embed(&self, prompt: &str) -> anyhow::Result<Vec<f32>> {
        for _ in 0..5 {
            if let Err(sleep) = self.rate_limiter.try_wait() {
                tokio::time::sleep(sleep).await;
                continue;
            }

            return self.embedding.embed(prompt).await;
        }

        anyhow::bail!("Rate limit exceeded for embedding computation");
    }
}
