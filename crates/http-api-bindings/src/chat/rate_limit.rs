use async_openai::{
    error::{ApiError, OpenAIError},
    types::{
        ChatCompletionResponseStream, CreateChatCompletionRequest, CreateChatCompletionResponse,
    },
};
use async_trait::async_trait;
use ratelimit::Ratelimiter;
use tabby_inference::ChatCompletionStream;

pub struct RateLimitedChatStream {
    completion: Box<dyn ChatCompletionStream>,
    rate_limiter: Ratelimiter,
}

impl RateLimitedChatStream {
    pub fn new(completion: Box<dyn ChatCompletionStream>, rate_limiter: Ratelimiter) -> Self {
        Self {
            completion,
            rate_limiter,
        }
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

        // Return an empty stream if the rate limit is exceeded
        Ok(Box::pin(futures::stream::empty()))
    }
}
