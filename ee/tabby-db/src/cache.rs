use std::{collections::HashMap, future::Future, hash::Hash, sync::Arc};

use chrono::{DateTime, Duration, Utc};
use tokio::sync::RwLock;

#[derive(Default)]
pub struct Cache<T> {
    value: RwLock<Option<T>>,
}

impl<T> Cache<T> {
    pub async fn new() -> Self {
        Cache {
            value: Default::default(),
        }
    }

    pub async fn invalidate(&self) {
        *self.value.write().await = None;
    }

    pub async fn get_or_refresh<F, E>(&self, refresh: impl Fn() -> F) -> Result<T, E>
    where
        T: Clone,
        F: Future<Output = Result<T, E>>,
    {
        let value = self.value.read().await;
        if let Some(value) = &*value {
            Ok(value.clone())
        } else {
            drop(value);
            let mut value = self.value.write().await;
            let generated = refresh().await?;
            *value = Some(generated.clone());
            Ok(generated)
        }
    }
}

pub struct TimedKeyedCache<K, V> {
    timeout: Duration,
    values: RwLock<HashMap<K, (DateTime<Utc>, V)>>,
}

impl<K, V> TimedKeyedCache<K, V> {
    pub fn new(timeout: Duration) -> Arc<Self>
    where
        K: Send + Sync + 'static,
        V: Send + Sync + 'static,
    {
        let cache = TimedKeyedCache {
            timeout,
            values: Default::default(),
        };
        let arc = Arc::new(cache);
        let clone = arc.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(timeout.to_std().unwrap()).await;
                let mut values = clone.values.write().await;
                let now = Utc::now();
                values.retain(|_k, (time, _v)| now - *time > clone.timeout);
            }
        });
        arc
    }

    pub async fn get_or_refresh<F, E>(&self, key: K, refresh: impl Fn(K) -> F) -> Result<V, E>
    where
        F: Future<Output = Result<V, E>>,
        K: Clone + Eq + PartialEq + Hash,
        V: Clone,
    {
        let inner_cache = self.values.read().await;
        let value = inner_cache.get(&key);
        if let Some((updated, value)) = value {
            if Utc::now() - *updated <= self.timeout {
                return Ok(value.clone());
            }
        };
        drop(inner_cache);
        let mut inner_cache = self.values.write().await;
        let new_val = refresh(key.clone()).await?;
        inner_cache.insert(key, (Utc::now(), new_val.clone()));
        Ok(new_val)
    }
}
