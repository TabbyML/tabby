use std::time::Duration;

use async_openai::{
    error::{ApiError, OpenAIError},
    types::{
        ChatCompletionResponseStream, CreateChatCompletionRequest, CreateChatCompletionResponse,
    },
};
use async_trait::async_trait;
use futures::stream::BoxStream;
use ratelimit::Ratelimiter;
use tabby_inference::{ChatCompletionStream, CompletionOptions, CompletionStream, Embedding};

fn new_rate_limiter(rpm: u64) -> Ratelimiter {
    Ratelimiter::builder(rpm, Duration::from_secs(60))
        .max_tokens(rpm)
        .initial_available(rpm)
        .build()
        .expect("Failed to create RateLimiter, please check the HttpModelConfig.rate_limit configuration")
}

pub struct RateLimitedEmbedding {
    embedding: Box<dyn Embedding>,
    rate_limiter: Ratelimiter,
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

pub struct RateLimitedCompletion {
    completion: Box<dyn CompletionStream>,
    rate_limiter: Ratelimiter,
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
        for _ in 0..5 {
            if let Err(sleep) = self.rate_limiter.try_wait() {
                tokio::time::sleep(sleep).await;
                continue;
            }

            return self.completion.generate(prompt, options).await;
        }

        // Return an empty stream if the rate limit is exceeded
        Box::pin(futures::stream::empty())
    }
}

pub struct RateLimitedChatStream {
    completion: Box<dyn ChatCompletionStream>,
    rate_limiter: Ratelimiter,
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
        for _ in 0..5 {
            if let Err(sleep) = self.rate_limiter.try_wait() {
                tokio::time::sleep(sleep).await;
                continue;
            }

            return self.completion.chat(request).await;
        }

        Err(OpenAIError::ApiError(ApiError {
            message: "Rate limit exceeded for chat completion".to_owned(),
            r#type: None,
            param: None,
            code: None,
        }))
    }

    async fn chat_stream(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        for _ in 0..5 {
            if let Err(sleep) = self.rate_limiter.try_wait() {
                tokio::time::sleep(sleep).await;
                continue;
            }

            return self.completion.chat_stream(request).await;
        }

        Err(OpenAIError::ApiError(ApiError {
            message: "Rate limit exceeded for chat completion".to_owned(),
            r#type: None,
            param: None,
            code: None,
        }))
    }
}
