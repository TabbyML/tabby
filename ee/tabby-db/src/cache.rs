use std::future::Future;

use tokio::sync::RwLock;

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

    pub async fn update(&self, f: impl FnOnce(&mut T)) {
        let mut lock = self.value.write().await;
        lock.as_mut().map(f);
    }

    pub async fn set(&self, value: T) {
        *self.value.write().await = Some(value);
    }
}
