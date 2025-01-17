use async_openai_alt::{
    error::OpenAIError,
    types::{
        ChatCompletionResponseStream, CreateChatCompletionRequest, CreateChatCompletionResponse,
    },
};
use async_trait::async_trait;
use futures::stream::BoxStream;
use leaky_bucket::RateLimiter;
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
}

pub fn new_embedding(embedding: Box<dyn Embedding>, request_per_minute: u64) -> impl Embedding {
    RateLimitedEmbedding {
        embedding,
        rate_limiter: new_rate_limiter(request_per_minute),
    }
}

#[async_trait]
impl Embedding for RateLimitedEmbedding {
    async fn embed(&self, prompt: &str) -> anyhow::Result<Vec<f32>> {
        self.rate_limiter.acquire(1).await;
        self.embedding
            .embed(prompt)
            .instrument(info_span!("rate_limited_compute_embedding"))
            .await
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
    async fn generate(&self, prompt: &str, options: CompletionOptions) -> BoxStream<String> {
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
