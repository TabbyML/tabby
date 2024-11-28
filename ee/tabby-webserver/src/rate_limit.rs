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
