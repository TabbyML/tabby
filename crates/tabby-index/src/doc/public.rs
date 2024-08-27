use std::sync::Arc;

use async_stream::stream;
use chrono::{DateTime, Utc};
use futures::StreamExt;
use tabby_common::index::corpus;
use tabby_inference::Embedding;

use super::create_web_builder;
use crate::{
    indexer::{IndexId, TantivyDocBuilder, ToIndexId},
    Indexer,
};

pub struct DocIndexer {
    builder: TantivyDocBuilder<WebDocument>,
    indexer: Indexer,
}

pub struct WebDocument {
    pub id: String,
    pub source_id: String,
    pub link: String,
    pub title: String,
    pub body: String,
}

impl ToIndexId for WebDocument {
    fn to_index_id(&self) -> IndexId {
        IndexId {
            source_id: self.source_id.clone(),
            id: self.id.clone(),
        }
    }
}

impl DocIndexer {
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        let builder = create_web_builder(embedding);
        let indexer = Indexer::new(corpus::WEB);
        Self { indexer, builder }
    }

    pub async fn add(&self, updated_at: DateTime<Utc>, document: WebDocument) -> bool {
        let is_document_empty = document.body.trim().is_empty();
        if is_document_empty || self.indexer.is_indexed_after(&document.id, updated_at) {
            return false;
        };

        stream! {
            let (id, s) = self.builder.build(document).await;
            self.indexer.delete(&id);

            for await doc in s.buffer_unordered(std::cmp::max(std::thread::available_parallelism().unwrap().get() * 2, 32)) {
                if let Ok(Some(doc)) = doc {
                    self.indexer.add(doc).await;
                }
            }
        }.count().await;
        true
    }

    pub fn commit(self) {
        self.indexer.commit();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use serial_test::serial;
    use tabby_common::path::set_tabby_root;
    use temp_testdir::TempDir;

    use super::*;

    struct FakeEmbedding;

    #[async_trait]
    impl Embedding for FakeEmbedding {
        async fn embed(&self, _prompt: &str) -> anyhow::Result<Vec<f32>> {
            Ok(vec![0.0; 16])
        }
    }

    fn create_testing_document() -> WebDocument {
        WebDocument {
            id: "1".to_string(),
            source_id: "1".to_string(),
            link: "https://example.com".to_string(),
            title: "Example".to_string(),
            body: "Hello, world!".to_string(),
        }
    }

    #[tokio::test]
    #[serial(set_tabby_root)]
    async fn test_add() {
        let tmp_dir = TempDir::default();
        set_tabby_root(tmp_dir.to_path_buf());
        let embedding = Arc::new(FakeEmbedding);
        let indexer = DocIndexer::new(embedding.clone());
        let updated_at = Utc::now();

        // Insert a new document
        assert!(indexer.add(updated_at, create_testing_document()).await);
        indexer.commit();

        // For document with the same id, and the updated_at is not newer, it should not be added.
        let indexer = DocIndexer::new(embedding);
        assert!(!indexer.add(updated_at, create_testing_document()).await);

        // For document with the same id, and the updated_at is newer, it should be added.
        assert!(
            indexer
                .add(
                    updated_at + chrono::Duration::seconds(1),
                    create_testing_document()
                )
                .await
        );
    }
}
