use std::sync::Arc;

use async_stream::stream;
use chrono::{DateTime, Utc};
use futures::StreamExt;
use tabby_common::index::corpus;
use tabby_inference::Embedding;

use super::{create_web_builder, SourceDocument};
use crate::{indexer::TantivyDocBuilder, Indexer};

pub struct DocIndexer {
    builder: TantivyDocBuilder<SourceDocument>,
    indexer: Indexer,
}

pub struct WebDocument {
    pub id: String,
    pub source_id: String,
    pub link: String,
    pub title: String,
    pub body: String,
}

impl From<WebDocument> for SourceDocument {
    fn from(value: WebDocument) -> Self {
        Self {
            id: value.id,
            source_id: value.source_id,
            link: value.link,
            title: value.title,
            body: value.body,
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
            let (id, s) = self.builder.build(document.into()).await;
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
