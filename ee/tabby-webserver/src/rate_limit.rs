use std::time::Duration;

use cached::{Cached, TimedCache};
use tokio::sync::Mutex;

pub struct UserRateLimiter {
    /// Mapping from user ID to rate limiter.
    rate_limiters: Mutex<TimedCache<String, ratelimit::Ratelimiter>>,
}

static USER_REQUEST_LIMIT_PER_MINUTE: u64 = 120;

impl Default for UserRateLimiter {
    fn default() -> Self {
        Self {
            // User rate limiter is hardcoded to 120 requests per minute, thus the timespan is 60 seconds.
            rate_limiters: Mutex::new(TimedCache::with_lifespan(60)),
        }
    }
}

impl UserRateLimiter {
    pub async fn is_allowed(&self, uri: &axum::http::Uri, user_id: &str) -> bool {
        // Do not limit health check requests.
        if uri.path().ends_with("/v1/health") || uri.path().ends_with("/v1beta/health") {
            return true;
        }

        let mut rate_limiters = self.rate_limiters.lock().await;
        let rate_limiter = rate_limiters.cache_get_or_set_with(user_id.to_string(), || {
            // Create a new rate limiter for this user.
            ratelimit::Ratelimiter::builder(USER_REQUEST_LIMIT_PER_MINUTE, Duration::from_secs(60))
                .max_tokens(USER_REQUEST_LIMIT_PER_MINUTE * 2)
                .initial_available(USER_REQUEST_LIMIT_PER_MINUTE)
                .build()
                .expect("Failed to create rate limiter")
        });
        if let Err(_sleep) = rate_limiter.try_wait() {
            // If the rate limiter is full, we return false.
            false
        } else {
            // If the rate limiter is not full, we return true.
            true
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[tokio::test]
    async fn test_user_rate_limiter() {
        let user_id = "test_user";
        let rate_limiter = UserRateLimiter::default();

        let uri: axum::http::Uri = "/v1/completions".parse().unwrap();
        let healthcheck_uri: axum::http::Uri = "/v1/health".parse().unwrap();

        // Test that the first `USER_REQUEST_LIMIT_PER_MINUTE` requests are allowed
        for _ in 0..USER_REQUEST_LIMIT_PER_MINUTE {
            assert!(rate_limiter.is_allowed(&uri, user_id).await);
        }

        // Test that the 201st request is not allowed
        assert!(!rate_limiter.is_allowed(&uri, user_id).await);

        // Test that health check requests are not limited
        assert!(rate_limiter.is_allowed(&healthcheck_uri, user_id).await);
    }
}
