use std::{sync::Arc, time::Duration};

use anyhow::Result;
use async_trait::async_trait;
use tabby_common::{
    api::doc::{DocSearch, DocSearchDocument, DocSearchError, DocSearchHit, DocSearchResponse},
    index::{self},
    path,
};
use tabby_inference::Embedding;
use tantivy::{
    collector::TopDocs,
    schema::{self, Value},
    Index, IndexReader, TantivyDocument,
};
use tokio::{
    sync::{Mutex, RwLock},
    time::sleep,
};
use tracing::{debug, warn};

struct DocSearchImpl {
    reader: IndexReader,
    embedding: Arc<dyn Embedding>,
}

impl DocSearchImpl {
    fn load(embedding: Arc<dyn Embedding>) -> Result<Self> {
        let index = Index::open_in_dir(path::doc_index_dir())?;

        Ok(Self {
            reader: index.reader_builder().try_into()?,
            embedding,
        })
    }

    async fn load_async(embedding: Arc<dyn Embedding>) -> DocSearchImpl {
        loop {
            match Self::load(embedding.clone()) {
                Ok(doc) => {
                    debug!("Index is ready, enabling doc search...");
                    return doc;
                }
                Err(err) => {
                    debug!("Doc index is not ready `{}`", err);
                }
            };

            sleep(Duration::from_secs(60)).await;
        }
    }
}

#[async_trait]
impl DocSearch for DocSearchImpl {
    async fn search(
        &self,
        q: &str,
        limit: usize,
        offset: usize,
    ) -> Result<DocSearchResponse, DocSearchError> {
        let schema = index::DocSearchSchema::instance();
        let embedding_tokens_query =
            schema.embedding_tokens_query(self.embedding.embed(q).await?.iter());

        let searcher = self.reader.searcher();
        let top_chunks = searcher.search(
            &embedding_tokens_query,
            &TopDocs::with_limit(limit).and_offset(offset),
        )?;

        let hits = top_chunks
            .iter()
            .filter_map(|(score, chunk_address)| {
                let chunk: TantivyDocument = searcher.doc(*chunk_address).ok()?;
                let doc_id = get_text(&chunk, schema.field_id);
                let chunk_text = get_text(&chunk, schema.field_chunk_text);

                let doc_query = schema.doc_query(doc_id);
                let top_docs = match searcher.search(&doc_query, &TopDocs::with_limit(1)) {
                    Err(err) => {
                        warn!("Failed to search doc `{}`: `{}`", doc_id, err);
                        return None;
                    }
                    Ok(top_docs) => top_docs,
                };
                let (_, doc_address) = top_docs.first()?;
                let doc: TantivyDocument = searcher.doc(*doc_address).ok()?;
                let title = get_text(&doc, schema.field_title);
                let link = get_text(&doc, schema.field_link);

                Some(DocSearchHit {
                    doc: DocSearchDocument {
                        title: title.to_string(),
                        link: link.to_string(),
                        snippet: chunk_text.to_string(),
                    },
                    score: *score,
                })
            })
            .collect();

        Ok(DocSearchResponse { hits })
    }
}

fn get_text(doc: &TantivyDocument, field: schema::Field) -> &str {
    doc.get_first(field).unwrap().as_str().unwrap()
}

struct DocSearchService {
    search: Arc<RwLock<Option<DocSearchImpl>>>,
    loader: tokio::task::JoinHandle<()>,
}

impl Drop for DocSearchService {
    fn drop(&mut self) {
        if !self.loader.is_finished() {
            self.loader.abort();
        }
    }
}

impl DocSearchService {
    fn new(embedding: Arc<dyn Embedding>) -> Self {
        let search = Arc::new(RwLock::new(None));
        let cloned_search = search.clone();
        let loader = tokio::spawn(async move {
            let doc = DocSearchImpl::load_async(embedding).await;
            *cloned_search.write().await = Some(doc);
        });

        Self {
            search: search.clone(),
            loader,
        }
    }
}

#[async_trait]
impl DocSearch for DocSearchService {
    async fn search(
        &self,
        q: &str,
        limit: usize,
        offset: usize,
    ) -> Result<DocSearchResponse, DocSearchError> {
        if let Some(imp) = self.search.read().await.as_ref() {
            imp.search(q, limit, offset).await
        } else {
            Err(DocSearchError::NotReady)
        }
    }
}

pub fn create(embedding: Arc<dyn Embedding>) -> impl DocSearch {
    DocSearchService::new(embedding)
}
