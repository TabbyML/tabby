use std::sync::Arc;

use anyhow::{bail, Result};
use async_stream::stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::json;
use tabby_common::{
    config::CodeRepository,
    index::{code, corpus},
};
use tabby_inference::Embedding;
use tokio::task::JoinHandle;
use tracing::{info_span, warn, Instrument};

use self::intelligence::SourceCode;
use crate::{
    code::intelligence::CodeIntelligence, indexer::TantivyDocBuilder, IndexAttributeBuilder,
};

//  Modules for creating code search index.
mod index;
pub mod intelligence;
mod languages;
mod repository;
mod types;

#[derive(Default)]
pub struct CodeIndexer {}

impl CodeIndexer {
    pub async fn refresh(
        &mut self,
        embedding: Arc<dyn Embedding>,
        repository: &CodeRepository,
    ) -> anyhow::Result<()> {
        logkit::info!(
            "Building source code index: {}",
            repository.canonical_git_url()
        );
        let commit = repository::sync_repository(repository)?;

        index::index_repository(embedding, repository, &commit).await;
        index::garbage_collection().await;

        Ok(())
    }

    pub async fn garbage_collection(&mut self, repositories: &[CodeRepository]) {
        repository::garbage_collection(repositories);
    }
}
struct CodeBuilder {
    embedding: Option<Arc<dyn Embedding>>,
}

impl CodeBuilder {
    fn new(embedding: Option<Arc<dyn Embedding>>) -> Self {
        Self { embedding }
    }
}

#[async_trait]
impl IndexAttributeBuilder<SourceCode> for CodeBuilder {
    async fn build_attributes(&self, source_code: &SourceCode) -> serde_json::Value {
        json!({
            code::fields::COMMIT: source_code.commit,
        })
    }

    async fn build_chunk_attributes<'a>(
        &self,
        source_code: &'a SourceCode,
    ) -> BoxStream<'a, JoinHandle<Result<(Vec<String>, serde_json::Value)>>> {
        let text = match source_code.read_content() {
            Ok(content) => content,
            Err(e) => {
                warn!(
                    "Failed to read content of '{}': {}",
                    source_code.filepath, e
                );

                return Box::pin(stream! {
                    let path = source_code.filepath.clone();
                    yield tokio::spawn(async move {
                        bail!("Failed to read content of '{}': {}", path, e);
                    });
                });
            }
        };

        let Some(embedding) = self.embedding.clone() else {
            warn!("No embedding service found for code indexing");
            return Box::pin(stream! {
                yield tokio::spawn(async move {
                    bail!("No embedding service found for code indexing");
                });
            });
        };

        let source_code = source_code.clone();
        let s = stream! {
            for await (start_line, body) in CodeIntelligence::chunks(&text, &source_code.language) {
                let mut attributes = json!({
                    code::fields::CHUNK_FILEPATH: source_code.filepath,
                    code::fields::CHUNK_GIT_URL: source_code.git_url,
                    code::fields::CHUNK_LANGUAGE: source_code.language,
                    code::fields::CHUNK_BODY: body,
                });

                // When text length is not equal to body length, it means this chunk is not the entire
                // content of the file, thus we need to record the start line.
                if text.len() != body.len() {
                    attributes[code::fields::CHUNK_START_LINE] = start_line.into();
                }
                let embedding = embedding.clone();
                let rewritten_body = format!("```{}\n{}\n```", source_code.filepath, body);
                yield tokio::spawn(async move {
                    match build_binarize_embedding_tokens(embedding.clone(), &rewritten_body).await {
                        Ok(tokens) => Ok((tokens, attributes)),
                        Err(err) => Err(err),
                    }
                });
            }
        };

        Box::pin(s)
    }
}

async fn build_binarize_embedding_tokens(
    embedding: Arc<dyn Embedding>,
    body: &str,
) -> Result<Vec<String>> {
    let embedding = match embedding
        .embed(body)
        .instrument(info_span!("index_compute_embedding", corpus = corpus::CODE))
        .await
    {
        Ok(x) => x,
        Err(err) => {
            bail!("Failed to embed chunk text: {}", err);
        }
    };

    let mut tokens = code::tokenize_code(body);
    for token in tabby_common::index::binarize_embedding(embedding.iter()) {
        tokens.push(token);
    }

    Ok(tokens)
}

pub fn create_code_builder(embedding: Option<Arc<dyn Embedding>>) -> TantivyDocBuilder<SourceCode> {
    let builder = CodeBuilder::new(embedding);
    TantivyDocBuilder::new(corpus::CODE, builder)
}
