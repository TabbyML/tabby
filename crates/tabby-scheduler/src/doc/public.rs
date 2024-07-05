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

    pub async fn add(&self, updated_at: DateTime<Utc>, document: WebDocument) {
        if let Some(dt) = self.indexer.read_updated_at(&document.id) {
            if dt >= updated_at {
                return;
            }
        };

        stream! {
            let (id, s) = self.builder.build(document).await;
            self.indexer.delete(&id);
            for await doc in s.buffer_unordered(std::cmp::max(std::thread::available_parallelism().unwrap().get() * 2, 32)) {
                if let Ok(Some(doc)) = doc {
                    self.indexer.add(doc).await;
                }
            }
        }.collect::<()>().await;
    }

    pub fn commit(self) {
        self.indexer.commit();
    }
}
