use std::{sync::Arc, time::Duration};

use tabby_common::{index::IndexSchema, path};
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

        if index.schema() != IndexSchema::instance().schema {
            return Err(anyhow::anyhow!("Index schema mismatch"));
        }

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
        let provider = Arc::new(RwLock::new(None));
        let cloned_provider = provider.clone();
        let loader = tokio::spawn(async move {
            let doc = Self::load_async().await;
            *cloned_provider.write().await = Some(doc);
        });

        Self { provider, loader }
    }
}

impl Drop for IndexReaderProvider {
    fn drop(&mut self) {
        self.loader.abort()
    }
}
