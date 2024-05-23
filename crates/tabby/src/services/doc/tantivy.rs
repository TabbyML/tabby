use std::{sync::Arc, time::Duration};

use anyhow::Result;
use async_trait::async_trait;
use tabby_common::{
    api::doc::{DocSearch, DocSearchDocument, DocSearchError, DocSearchHit, DocSearchResponse},
    index::{self, doc},
    path,
};
use tabby_inference::Embedding;
use tantivy::{
    collector::TopDocs,
    schema::{self, document::ReferenceValue, Value},
    Index, IndexReader, TantivyDocument,
};
use tokio::{sync::RwLock, time::sleep};
use tracing::{debug, warn};

struct DocSearchImpl {
    reader: IndexReader,
    embedding: Arc<dyn Embedding>,
}

impl DocSearchImpl {
    fn load(embedding: Arc<dyn Embedding>) -> Result<Self> {
        let index = Index::open_in_dir(path::index_dir())?;

        Ok(Self {
            reader: index.reader_builder().try_into()?,
            embedding,
        })
    }

    async fn load_async(embedding: Arc<dyn Embedding>) -> DocSearchImpl {
        loop {
            if let Ok(doc) = Self::load(embedding.clone()) {
                debug!("Index is ready, enabling doc search...");
                return doc;
            }

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
        let schema = index::IndexSchema::instance();
        let embedding = self.embedding.embed(q).await?;
        let embedding_tokens_query =
            index::embedding_tokens_query(embedding.len(), embedding.iter());

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
                let chunk_text = get_json_text_field(
                    &chunk,
                    schema.field_chunk_attributes,
                    doc::fields::CHUNK_TEXT,
                );

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
                let title = get_json_text_field(&doc, schema.field_attributes, doc::fields::TITLE);
                let link = get_json_text_field(&doc, schema.field_attributes, doc::fields::LINK);

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

fn get_json_text_field<'a>(doc: &'a TantivyDocument, field: schema::Field, name: &str) -> &'a str {
    let ReferenceValue::Object(obj) = doc.get_first(field).unwrap() else {
        panic!("Field {} is not an object", name);
    };
    obj.into_iter()
        .find(|(k, _)| *k == name)
        .unwrap()
        .1
        .as_str()
        .unwrap()
}

pub struct DocSearchService {
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
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
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
