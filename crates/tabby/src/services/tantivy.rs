use std::{sync::Arc, time::Duration};

use tabby_common::path;
use tantivy::{Index, IndexReader};
use tokio::sync::RwLock;
use tracing::debug;

pub struct IndexReaderProvider {
    provider: Arc<RwLock<Option<IndexReader>>>,
    loader: tokio::task::JoinHandle<()>,
}

impl IndexReaderProvider {
    pub fn reader(
        &self,
    ) -> impl futures::Future<Output = tokio::sync::RwLockReadGuard<Option<IndexReader>>> {
        self.provider.read()
    }

    fn load() -> anyhow::Result<IndexReader> {
        let index = Index::open_in_dir(path::index_dir())?;

        Ok(index.reader_builder().try_into()?)
    }

    async fn load_async() -> IndexReader {
        loop {
            if let Ok(provider) = Self::load() {
                debug!("Index is ready, enabling search...");
                return provider;
            }

            tokio::time::sleep(Duration::from_secs(60)).await;
        }
    }
}

impl Default for IndexReaderProvider {
    fn default() -> Self {
        let search = Arc::new(RwLock::new(None));
        let cloned_search = search.clone();
        let loader = tokio::spawn(async move {
            let doc = Self::load_async().await;
            *cloned_search.write().await = Some(doc);
        });

        Self {
            provider: search.clone(),
            loader,
        }
    }
}

impl Drop for IndexReaderProvider {
    fn drop(&mut self) {
        self.loader.abort()
    }
}
