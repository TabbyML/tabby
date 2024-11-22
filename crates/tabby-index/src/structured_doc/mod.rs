pub mod public;
mod types;

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::json;
use tabby_common::index::{corpus, structured_doc};
use tabby_inference::Embedding;
use tokio::task::JoinHandle;
use types::{BuildStructuredDoc, StructuredDoc};

use crate::{indexer::TantivyDocBuilder, IndexAttributeBuilder};

pub struct StructuredDocBuilder {
    embedding: Arc<dyn Embedding>,
}

impl StructuredDocBuilder {
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        Self { embedding }
    }
}

#[async_trait]
impl IndexAttributeBuilder<StructuredDoc> for StructuredDocBuilder {
    async fn build_attributes(&self, document: &StructuredDoc) -> serde_json::Value {
        let mut attributes = document.build_attributes().await;
        attributes
            .as_object_mut()
            .unwrap()
            .insert(structured_doc::fields::KIND.into(), json!(document.kind()));
        attributes
    }

    async fn build_chunk_attributes<'a>(
        &self,
        document: &'a StructuredDoc,
    ) -> BoxStream<'a, JoinHandle<(Vec<String>, serde_json::Value)>> {
        let embedding = self.embedding.clone();
        document.build_chunk_attributes(embedding).await
    }
}

fn create_structured_doc_builder(
    embedding: Arc<dyn Embedding>,
) -> TantivyDocBuilder<StructuredDoc> {
    let builder = StructuredDocBuilder::new(embedding);
    TantivyDocBuilder::new(corpus::STRUCTURED_DOC, builder)
}
