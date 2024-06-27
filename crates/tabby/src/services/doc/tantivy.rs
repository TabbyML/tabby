use std::{collections::HashSet, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use tabby_common::{
    api::doc::{DocSearch, DocSearchDocument, DocSearchError, DocSearchHit, DocSearchResponse},
    index::{self, corpus, doc},
};
use tabby_inference::Embedding;
use tantivy::{
    collector::TopDocs,
    query::{BooleanQuery, ConstScoreQuery, Occur},
    schema::{self, Value},
    IndexReader, TantivyDocument,
};
use tracing::warn;

use crate::services::tantivy::IndexReaderProvider;

struct DocSearchImpl {
    embedding: Arc<dyn Embedding>,
}

const EMBEDDING_SCORE_THRESHOLD: f32 = 0.75;

impl DocSearchImpl {
    fn new(embedding: Arc<dyn Embedding>) -> Self {
        Self { embedding }
    }

    async fn search(
        &self,
        reader: &IndexReader,
        q: &str,
        limit: usize,
    ) -> Result<DocSearchResponse, DocSearchError> {
        let schema = index::IndexSchema::instance();
        let embedding = self.embedding.embed(q).await?;
        let embedding_tokens_query =
            index::embedding_tokens_query(embedding.len(), embedding.iter());
        let corpus_query = schema.corpus_query(corpus::WEB);
        let query = BooleanQuery::new(vec![
            (
                Occur::Must,
                Box::new(ConstScoreQuery::new(corpus_query, 0.0)),
            ),
            (Occur::Must, Box::new(embedding_tokens_query)),
        ]);

        let searcher = reader.searcher();
        let top_chunks = searcher.search(&query, &TopDocs::with_limit(limit * 2))?;

        let chunks = {
            // Extract all chunks.
            let mut chunks: Vec<_> = top_chunks
                .iter()
                .filter_map(|(score, chunk_address)| {
                    let chunk: TantivyDocument = searcher.doc(*chunk_address).ok()?;
                    let doc_id = get_text(&chunk, schema.field_id).to_owned();
                    Some(ScoredChunk {
                        score: *score,
                        chunk,
                        doc_id,
                    })
                })
                .collect();

            // Sort by score in descending order.
            chunks.sort_unstable_by(|lhs, rhs| rhs.score.total_cmp(&lhs.score));

            // Deduplicate by doc_id.
            let mut doc_ids = HashSet::new();
            chunks.retain(|x| doc_ids.insert(x.doc_id.clone()));

            chunks
        };

        let hits = chunks
            .iter()
            .filter_map(
                |ScoredChunk {
                     doc_id,
                     score,
                     chunk,
                 }| {
                    let chunk_text = get_json_text_field(
                        chunk,
                        schema.field_chunk_attributes,
                        doc::fields::CHUNK_TEXT,
                    );

                    let doc_query = schema.doc_query(corpus::WEB, doc_id);
                    let top_docs = match searcher.search(&doc_query, &TopDocs::with_limit(1)) {
                        Err(err) => {
                            warn!("Failed to search doc `{}`: `{}`", doc_id, err);
                            return None;
                        }
                        Ok(top_docs) => top_docs,
                    };
                    let (_, doc_address) = top_docs.first()?;
                    let doc: TantivyDocument = searcher.doc(*doc_address).ok()?;
                    let title =
                        get_json_text_field(&doc, schema.field_attributes, doc::fields::TITLE);
                    let link =
                        get_json_text_field(&doc, schema.field_attributes, doc::fields::LINK);

                    Some(DocSearchHit {
                        doc: DocSearchDocument {
                            title: title.to_string(),
                            link: link.to_string(),
                            snippet: chunk_text.to_string(),
                        },
                        score: *score,
                    })
                },
            )
            .filter(|x| x.score >= EMBEDDING_SCORE_THRESHOLD)
            .take(limit)
            .collect();

        Ok(DocSearchResponse { hits })
    }
}

fn get_text(doc: &TantivyDocument, field: schema::Field) -> &str {
    doc.get_first(field).unwrap().as_str().unwrap()
}

fn get_json_text_field<'a>(doc: &'a TantivyDocument, field: schema::Field, name: &str) -> &'a str {
    doc.get_first(field)
        .unwrap()
        .as_object()
        .unwrap()
        .find(|(k, _)| *k == name)
        .unwrap()
        .1
        .as_str()
        .unwrap()
}

struct ScoredChunk {
    doc_id: String,
    score: f32,
    chunk: TantivyDocument,
}

pub struct DocSearchService {
    imp: DocSearchImpl,
    provider: Arc<IndexReaderProvider>,
}

impl DocSearchService {
    pub fn new(embedding: Arc<dyn Embedding>, provider: Arc<IndexReaderProvider>) -> Self {
        Self {
            imp: DocSearchImpl::new(embedding),
            provider,
        }
    }
}

#[async_trait]
impl DocSearch for DocSearchService {
    async fn search(&self, q: &str, limit: usize) -> Result<DocSearchResponse, DocSearchError> {
        if let Some(reader) = self.provider.reader().await.as_ref() {
            self.imp.search(reader, q, limit).await
        } else {
            Err(DocSearchError::NotReady)
        }
    }
}
