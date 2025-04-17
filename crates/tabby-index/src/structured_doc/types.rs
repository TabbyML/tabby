pub mod commit;
pub mod ingested;
pub mod issue;
pub mod page;
pub mod pull;
pub mod web;

use std::sync::Arc;

use anyhow::{bail, Result};
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

pub const KIND_WEB: &str = "web";
pub const KIND_ISSUE: &str = "issue";
pub const KIND_PULL: &str = "pull";
pub const KIND_COMMIT: &str = "commit";
pub const KIND_PAGE: &str = "page";
pub const KIND_INGESTED: &str = "ingested";

impl StructuredDoc {
    pub fn id(&self) -> &str {
        match &self.fields {
            StructuredDocFields::Web(web) => &web.link,
            StructuredDocFields::Issue(issue) => &issue.link,
            StructuredDocFields::Pull(pull) => &pull.link,
            StructuredDocFields::Commit(commit) => &commit.sha,
            StructuredDocFields::Page(page) => &page.link,
            StructuredDocFields::Ingested(ingested) => &ingested.id,
        }
    }

    pub fn kind(&self) -> &'static str {
        match &self.fields {
            StructuredDocFields::Web(_) => KIND_WEB,
            StructuredDocFields::Issue(_) => KIND_ISSUE,
            StructuredDocFields::Pull(_) => KIND_PULL,
            StructuredDocFields::Commit(_) => KIND_COMMIT,
            StructuredDocFields::Page(_) => KIND_PAGE,
            StructuredDocFields::Ingested(_) => KIND_INGESTED,
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
    ) -> BoxStream<JoinHandle<Result<(Vec<String>, serde_json::Value)>>>;
}

pub enum StructuredDocFields {
    Web(web::WebDocument),
    Issue(issue::IssueDocument),
    Pull(pull::PullDocument),
    Commit(commit::CommitDocument),
    Page(page::PageDocument),
    Ingested(ingested::IngestedDocument),
}

#[async_trait]
impl BuildStructuredDoc for StructuredDoc {
    fn should_skip(&self) -> bool {
        match &self.fields {
            StructuredDocFields::Web(doc) => doc.should_skip(),
            StructuredDocFields::Issue(doc) => doc.should_skip(),
            StructuredDocFields::Pull(doc) => doc.should_skip(),
            StructuredDocFields::Commit(doc) => doc.should_skip(),
            StructuredDocFields::Page(doc) => doc.should_skip(),
            StructuredDocFields::Ingested(doc) => doc.should_skip(),
        }
    }

    async fn build_attributes(&self) -> serde_json::Value {
        match &self.fields {
            StructuredDocFields::Web(doc) => doc.build_attributes().await,
            StructuredDocFields::Issue(doc) => doc.build_attributes().await,
            StructuredDocFields::Pull(doc) => doc.build_attributes().await,
            StructuredDocFields::Commit(doc) => doc.build_attributes().await,
            StructuredDocFields::Page(doc) => doc.build_attributes().await,
            StructuredDocFields::Ingested(doc) => doc.build_attributes().await,
        }
    }

    async fn build_chunk_attributes(
        &self,
        embedding: Arc<dyn Embedding>,
    ) -> BoxStream<JoinHandle<Result<(Vec<String>, serde_json::Value)>>> {
        match &self.fields {
            StructuredDocFields::Web(doc) => doc.build_chunk_attributes(embedding).await,
            StructuredDocFields::Issue(doc) => doc.build_chunk_attributes(embedding).await,
            StructuredDocFields::Pull(doc) => doc.build_chunk_attributes(embedding).await,
            StructuredDocFields::Commit(doc) => doc.build_chunk_attributes(embedding).await,
            StructuredDocFields::Page(doc) => doc.build_chunk_attributes(embedding).await,
            StructuredDocFields::Ingested(doc) => doc.build_chunk_attributes(embedding).await,
        }
    }
}

async fn build_tokens(embedding: Arc<dyn Embedding>, text: &str) -> Result<Vec<String>> {
    let embedding = match embedding.embed(text).await {
        Ok(embedding) => embedding,
        Err(err) => {
            warn!("Failed to embed chunk text: {}", err);
            bail!("Failed to embed chunk text: {}", err);
        }
    };

    let mut chunk_embedding_tokens = vec![];
    for token in tabby_common::index::binarize_embedding(embedding.iter()) {
        chunk_embedding_tokens.push(token);
    }

    Ok(chunk_embedding_tokens)
}
