use std::time::Duration;

use cached::{Cached, TimedCache};
use tokio::sync::Mutex;

pub struct UserRateLimiter {
    /// Mapping from user ID to rate limiter.
    rate_limiters: Mutex<TimedCache<String, ratelimit::Ratelimiter>>,
}

static USER_REQUEST_LIMIT_PER_MINUTE: u64 = 200;

impl Default for UserRateLimiter {
    fn default() -> Self {
        Self {
            // User rate limiter is hardcoded to 200 requests per minute, thus the timespan is 60 seconds.
            rate_limiters: Mutex::new(TimedCache::with_lifespan(60)),
        }
    }
}

impl UserRateLimiter {
    pub async fn is_allowed(&self, user_id: &str) -> bool {
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
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_user_rate_limiter() {
        let user_id = "test_user";
        let rate_limiter = UserRateLimiter::default();

        // Test that the first 200 requests are allowed
        for _ in 0..USER_REQUEST_LIMIT_PER_MINUTE {
            assert!(rate_limiter.is_allowed(user_id).await);
        }

        // Test that the 201st request is not allowed
        assert!(!rate_limiter.is_allowed(user_id).await);
    }
}
