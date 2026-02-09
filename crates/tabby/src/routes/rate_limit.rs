use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use leaky_bucket::RateLimiter;
use tokio::time::Duration;

/// Creates a rate limiter for the given requests per minute.
fn new_rate_limiter(rpm: u32) -> RateLimiter {
    let rps = (rpm as f64 / 60.0).ceil() as usize;
    RateLimiter::builder()
        .initial(rps)
        .interval(Duration::from_secs(1))
        .refill(rps)
        .build()
}

/// Key for rate limiting: (endpoint_name, user_id)
type RateLimitKey = (String, String);

/// Shared state for rate limiters, keyed by (endpoint_name, user_id)
#[derive(Default)]
pub struct EndpointRateLimiters {
    limiters: RwLock<HashMap<RateLimitKey, Arc<RateLimiter>>>,
}

impl EndpointRateLimiters {
    pub fn new() -> Self {
        Self {
            limiters: RwLock::new(HashMap::new()),
        }
    }

    /// Get or create a rate limiter for the given endpoint and user.
    pub fn get_or_create(&self, endpoint_name: &str, user_id: &str, rpm: u32) -> Arc<RateLimiter> {
        let key = (endpoint_name.to_string(), user_id.to_string());

        // Try read lock first
        {
            let limiters = self.limiters.read().unwrap();
            if let Some(limiter) = limiters.get(&key) {
                return limiter.clone();
            }
        }

        // Need to create a new limiter, acquire write lock
        let mut limiters = self.limiters.write().unwrap();
        // Double-check in case another thread created it
        if let Some(limiter) = limiters.get(&key) {
            return limiter.clone();
        }

        let limiter = Arc::new(new_rate_limiter(rpm));
        limiters.insert(key, limiter.clone());
        limiter
    }
}
