use async_trait::async_trait;
use futures::stream::BoxStream;
use ratelimit::Ratelimiter;
use tabby_inference::{CompletionOptions, CompletionStream};

pub struct RateLimitedCompletion {
    completion: Box<dyn CompletionStream>,
    rate_limiter: Ratelimiter,
}

impl RateLimitedCompletion {
    pub fn new(completion: Box<dyn CompletionStream>, rate_limiter: Ratelimiter) -> Self {
        Self {
            completion,
            rate_limiter,
        }
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
