mod issue;
mod web;

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use tabby_inference::Embedding;
use tokio::task::JoinHandle;
use tracing::warn;

use crate::indexer::{IndexId, ToIndexId};

pub struct StructuredDoc {
    pub source_id: String,
    pub fields: StructuredDocFields,
}

impl StructuredDoc {
    pub fn id(&self) -> &str {
        match &self.fields {
            StructuredDocFields::Web(web) => &web.link,
            StructuredDocFields::Issue(issue) => &issue.link,
        }
    }

    pub fn kind(&self) -> &'static str {
        match &self.fields {
            StructuredDocFields::Web(_) => "web",
            StructuredDocFields::Issue(_) => "issue",
        }
    }
}

impl ToIndexId for StructuredDoc {
    fn to_index_id(&self) -> IndexId {
        IndexId {
            source_id: self.source_id.clone(),
            id: self.id().to_owned(),
        }
    }
}

#[async_trait]
pub trait BuildStructuredDoc {
    fn should_skip(&self) -> bool;

    async fn build_attributes(&self) -> serde_json::Value;
    async fn build_chunk_attributes(
        &self,
        embedding: Arc<dyn Embedding>,
    ) -> BoxStream<JoinHandle<(Vec<String>, serde_json::Value)>>;
}

pub enum StructuredDocFields {
    Web(web::WebDocument),
    Issue(issue::IssueDocument),
}

#[async_trait]
impl BuildStructuredDoc for StructuredDoc {
    fn should_skip(&self) -> bool {
        match &self.fields {
            StructuredDocFields::Web(doc) => doc.should_skip(),
            StructuredDocFields::Issue(doc) => doc.should_skip(),
        }
    }

    async fn build_attributes(&self) -> serde_json::Value {
        match &self.fields {
            StructuredDocFields::Web(doc) => doc.build_attributes().await,
            StructuredDocFields::Issue(doc) => doc.build_attributes().await,
        }
    }

    async fn build_chunk_attributes(
        &self,
        embedding: Arc<dyn Embedding>,
    ) -> BoxStream<JoinHandle<(Vec<String>, serde_json::Value)>> {
        match &self.fields {
            StructuredDocFields::Web(doc) => doc.build_chunk_attributes(embedding).await,
            StructuredDocFields::Issue(doc) => doc.build_chunk_attributes(embedding).await,
        }
    }
}

async fn build_tokens(embedding: Arc<dyn Embedding>, text: &str) -> Vec<String> {
    let embedding = match embedding.embed(text).await {
        Ok(embedding) => embedding,
        Err(err) => {
            warn!("Failed to embed chunk text: {}", err);
            return vec![];
        }
    };

    let mut chunk_embedding_tokens = vec![];
    for token in tabby_common::index::binarize_embedding(embedding.iter()) {
        chunk_embedding_tokens.push(token);
    }

    chunk_embedding_tokens
}
