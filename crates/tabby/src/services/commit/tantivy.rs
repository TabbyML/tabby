use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use tabby_common::{
    api::{
        commit::{
            CommitHistoryDocument, CommitHistorySearch, CommitHistorySearchHit,
            CommitHistorySearchResponse,
        },
        Result, SearchError,
    },
    index::{self, corpus},
};
use tabby_inference::Embedding;
use tantivy::{
    collector::TopDocs,
    query::{BooleanQuery, ConstScoreQuery, Occur, Query},
    schema::{self, Value},
    IndexReader, TantivyDocument,
};
use tracing::warn;

use crate::services::tantivy::IndexReaderProvider;

const EMBEDDING_SCORE_THRESHOLD: f32 = 0.75;

pub struct CommitHistorySearchImpl {
    embedding: Arc<dyn Embedding>,
    provider: Arc<IndexReaderProvider>,
}

pub fn new(
    embedding: Arc<dyn Embedding>,
    provider: Arc<IndexReaderProvider>,
) -> CommitHistorySearchImpl {
    CommitHistorySearchImpl {
        embedding,
        provider,
    }
}

#[async_trait]
impl CommitHistorySearch for CommitHistorySearchImpl {
    async fn search(
        &self,
        source_id: &str,
        content: &str,
        limit: usize,
    ) -> Result<CommitHistorySearchResponse> {
        if let Some(reader) = self.provider.reader().await.as_ref() {
            self.index_search(reader, source_id, content, limit).await
        } else {
            Err(SearchError::NotReady)
        }
    }
}

struct ScoredChunk {
    doc_id: String,
    score: f32,
    chunk: TantivyDocument,
}

impl CommitHistorySearchImpl {
    async fn index_search(
        &self,
        reader: &IndexReader,
        source_id: &str,
        content: &str,
        limit: usize,
    ) -> Result<CommitHistorySearchResponse> {
        let schema = index::IndexSchema::instance();
        let query = {
            let embedding = self.embedding.embed(content).await?;
            let embedding_tokens_query =
                index::embedding_tokens_query(embedding.len(), embedding.iter());
            let corpus_query = schema.corpus_query(corpus::COMMIT_HISTORY);

            let mut query_clauses: Vec<(Occur, Box<dyn Query>)> = vec![
                (
                    Occur::Must,
                    Box::new(ConstScoreQuery::new(corpus_query, 0.0)),
                ),
                (Occur::Must, Box::new(embedding_tokens_query)),
            ];

            if !source_id.is_empty() {
                let source_id_query = Box::new(schema.source_id_query(source_id));
                let source_id_query = ConstScoreQuery::new(source_id_query, 0.0);
                query_clauses.push((Occur::Must, Box::new(source_id_query)));
            }
            BooleanQuery::new(query_clauses)
        };

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
                    let doc_query = schema.doc_query(corpus::COMMIT_HISTORY, doc_id);
                    let top_docs = match searcher.search(&doc_query, &TopDocs::with_limit(1)) {
                        Err(err) => {
                            warn!("Failed to search doc `{}`: `{}`", doc_id, err);
                            return None;
                        }
                        Ok(top_docs) => top_docs,
                    };
                    let (_, doc_address) = top_docs.first()?;
                    let doc: TantivyDocument = searcher.doc(*doc_address).ok()?;
                    CommitHistoryDocument::from_tantivy_document(&doc, chunk).map(|commit| {
                        CommitHistorySearchHit {
                            score: *score,
                            commit,
                        }
                    })
                },
            )
            .filter(|x| x.score >= EMBEDDING_SCORE_THRESHOLD)
            .take(limit)
            .collect();

        Ok(CommitHistorySearchResponse { hits })
    }
}

fn get_text(doc: &TantivyDocument, field: schema::Field) -> &str {
    doc.get_first(field).unwrap().as_str().unwrap()
}
