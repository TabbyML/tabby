use async_openai_alt::{
    error::OpenAIError,
    types::{
        ChatCompletionResponseStream, CreateChatCompletionRequest, CreateChatCompletionResponse,
    },
};
use async_trait::async_trait;
use futures::stream::BoxStream;
use leaky_bucket::RateLimiter;
use llm_resilience::{CircuitBreaker, CircuitBreakerError};
use tabby_inference::{ChatCompletionStream, CompletionOptions, CompletionStream, Embedding};
use tokio::time::Duration;
use tracing::{info_span, Instrument};

fn new_rate_limiter(rpm: u64) -> RateLimiter {
    let rps = (rpm as f64 / 60.0).ceil() as usize;
    RateLimiter::builder()
        .initial(rps)
        .interval(Duration::from_secs(1))
        .refill(rps)
        .build()
}

pub struct RateLimitedEmbedding {
    embedding: Box<dyn Embedding>,
    rate_limiter: RateLimiter,
    circuit_breaker: CircuitBreaker,
}

pub fn new_embedding(embedding: Box<dyn Embedding>, request_per_minute: u64) -> impl Embedding {
    RateLimitedEmbedding {
        embedding,
        rate_limiter: new_rate_limiter(request_per_minute),
        // Stop sending index chunks to an embedding server that is down:
        // open after 5 consecutive failures, probe again after 30s.
        circuit_breaker: CircuitBreaker::new(5, 0.8, Duration::from_secs(30)),
    }
}

#[async_trait]
impl Embedding for RateLimitedEmbedding {
    async fn embed(&self, prompt: &str) -> anyhow::Result<Vec<f32>> {
        self.circuit_breaker
            .call(|| async {
                self.rate_limiter.acquire(1).await;
                self.embedding
                    .embed(prompt)
                    .instrument(info_span!("rate_limited_compute_embedding"))
                    .await
            })
            .await
            .map_err(|e| match e {
                CircuitBreakerError::Open => {
                    anyhow::anyhow!("embedding endpoint is failing, circuit breaker open")
                }
                CircuitBreakerError::Failed(e) => e,
            })
    }
}

pub struct RateLimitedCompletion {
    completion: Box<dyn CompletionStream>,
    rate_limiter: RateLimiter,
}

pub fn new_completion(
    completion: Box<dyn CompletionStream>,
    request_per_minute: u64,
) -> impl CompletionStream {
    RateLimitedCompletion {
        completion,
        rate_limiter: new_rate_limiter(request_per_minute),
    }
}

#[async_trait]
impl CompletionStream for RateLimitedCompletion {
    async fn generate(
        &self,
        prompt: &str,
        options: CompletionOptions,
    ) -> BoxStream<'life0, String> {
        self.rate_limiter.acquire(1).await;
        self.completion.generate(prompt, options).await
    }
}

pub struct RateLimitedChatStream {
    completion: Box<dyn ChatCompletionStream>,
    rate_limiter: RateLimiter,
}

pub fn new_chat(
    completion: Box<dyn ChatCompletionStream>,
    request_per_minute: u64,
) -> impl ChatCompletionStream {
    RateLimitedChatStream {
        completion,
        rate_limiter: new_rate_limiter(request_per_minute),
    }
}

#[async_trait]
impl ChatCompletionStream for RateLimitedChatStream {
    async fn chat(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, OpenAIError> {
        self.rate_limiter.acquire(1).await;
        self.completion.chat(request).await
    }

    async fn chat_stream(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        self.rate_limiter.acquire(1).await;
        self.completion.chat_stream(request).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use super::*;

    struct FailingEmbedding(Arc<AtomicUsize>);

    #[async_trait]
    impl Embedding for FailingEmbedding {
        async fn embed(&self, _prompt: &str) -> anyhow::Result<Vec<f32>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Err(anyhow::anyhow!("connection refused"))
        }
    }

    #[tokio::test]
    async fn test_embedding_stops_calling_failing_endpoint() {
        let calls = Arc::new(AtomicUsize::new(0));
        let embedding = new_embedding(Box::new(FailingEmbedding(calls.clone())), 6000);

        for _ in 0..20 {
            assert!(embedding.embed("fn main() {}").await.is_err());
        }

        assert_eq!(calls.load(Ordering::SeqCst), 5);
    }
}
