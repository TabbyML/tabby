use std::collections::HashSet;

use anyhow::{bail, Result};
use async_stream::stream;
use futures::{stream::BoxStream, Stream, StreamExt};
use serde_json::json;
use tabby_common::{
    index::{structured_doc::fields::KIND, IndexSchema, FIELD_SOURCE_ID},
    path,
};
use tantivy::{
    aggregation::{
        agg_req::Aggregation,
        agg_result::{AggregationResult, BucketResult},
        AggregationCollector, Key,
    },
    collector::TopDocs,
    doc,
    query::AllQuery,
    schema::{self, document::CompactDocValue, Value},
    DateTime, DocAddress, DocSet, IndexWriter, Searcher, TantivyDocument, Term, TERMINATED,
};
use tokio::{sync::mpsc, task::JoinHandle};
use tracing::{debug, warn};

use crate::tantivy_utils::open_or_create_index;

pub struct IndexId {
    pub source_id: String,
    pub id: String,
}

pub trait ToIndexId {
    fn to_index_id(&self) -> IndexId;
}

#[async_trait::async_trait]
pub trait IndexAttributeBuilder<T>: Send + Sync {
    /// Build document level attributes, these attributes are only stored but not indexed.
    async fn build_attributes(&self, document: &T) -> serde_json::Value;

    /// Build chunk level attributes, these attributes are stored and indexed.
    async fn build_chunk_attributes<'a>(
        &self,
        document: &'a T,
    ) -> BoxStream<'a, JoinHandle<Result<(Vec<String>, serde_json::Value)>>>;
}

pub struct TantivyDocBuilder<T> {
    corpus: &'static str,
    builder: Box<dyn IndexAttributeBuilder<T>>,
}

impl<T: ToIndexId> TantivyDocBuilder<T> {
    pub fn new(corpus: &'static str, builder: impl IndexAttributeBuilder<T> + 'static) -> Self {
        Self {
            corpus,
            builder: Box::new(builder),
        }
    }

    pub async fn build(
        &self,
        document: T,
    ) -> (
        String,
        impl Stream<Item = JoinHandle<Option<TantivyDocument>>> + '_,
    ) {
        let schema = IndexSchema::instance();
        let IndexId { source_id, id } = document.to_index_id();

        let now = tantivy::time::OffsetDateTime::now_utc();
        let updated_at = tantivy::DateTime::from_utc(now);

        let cloned_id = id.clone();
        let doc_id = id.clone();
        let doc_attributes = self.builder.build_attributes(&document).await;
        let s = stream! {
            let (tx, mut rx) = mpsc::channel(32);

            for await chunk_doc in self.build_chunks(cloned_id, source_id.clone(), updated_at, document).await {
                let tx = tx.clone();
                let doc_id = doc_id.clone();
                yield tokio::spawn(async move {
                    match chunk_doc.await {
                        Ok(Ok(doc)) => {
                            Some(doc)
                        }
                        Ok(Err(e)) => {
                            warn!("Failed to build chunk for document '{}': {}", doc_id, e);
                            tx.send(()).await.unwrap_or_else(|e| {
                                warn!("Failed to send error signal for document '{}': {}", doc_id, e);
                            });
                            None
                        }
                        Err(e) => {
                            warn!("Failed to call build chunk '{}': {}", doc_id, e);
                            tx.send(()).await.unwrap_or_else(|e| {
                                warn!("Failed to send error signal for document '{}': {}", doc_id, e);
                            });
                            None
                        }
                    }
                });
            };

            // drop tx to signal the end of the stream
            // the cloned is dropped in its own thread
            drop(tx);

            let mut doc = doc! {
                schema.field_id => doc_id,
                schema.field_source_id => source_id,
                schema.field_corpus => self.corpus,
                schema.field_attributes => doc_attributes,
                schema.field_updated_at => updated_at,
            };

            yield tokio::spawn(async move {
                let mut failed_count = 0;
                while (rx.recv().await).is_some() {
                    failed_count += 1;
                }
                if failed_count > 0 {
                    doc.add_u64(schema.field_failed_chunks_count, failed_count as u64);
                }
                Some(doc)
             });
        };

        (id, s)
    }

    async fn build_chunks(
        &self,
        id: String,
        source_id: String,
        updated_at: tantivy::DateTime,
        document: T,
    ) -> impl Stream<Item = JoinHandle<Result<TantivyDocument>>> + '_ {
        let kind = self.corpus;
        stream! {
            let schema = IndexSchema::instance();
            for await (chunk_id, task) in self.builder.build_chunk_attributes(&document).await.enumerate() {
                let id = id.clone();
                let source_id = source_id.clone();

                yield tokio::spawn(async move {
                    let built_chunk_attributes_result = task.await?;
                    let (tokens, chunk_attributes) = built_chunk_attributes_result?;

                    let mut doc = doc! {
                        schema.field_id => id,
                        schema.field_source_id => source_id,
                        schema.field_corpus => kind,
                        schema.field_updated_at => updated_at,
                        schema.field_chunk_id => format!("{}-{}", id, chunk_id),
                        schema.field_chunk_attributes => chunk_attributes,
                    };

                    for token in &tokens {
                        doc.add_text(schema.field_chunk_tokens, token);
                    }

                    Ok(doc)
                });
            }
        }
    }

    pub async fn backfill_doc_attributes(
        &self,
        origin: &TantivyDocument,
        doc: &T,
    ) -> TantivyDocument {
        let schema = IndexSchema::instance();
        let mut doc = doc! {
            schema.field_id => get_text(origin, schema.field_id),
            schema.field_source_id => get_text(origin, schema.field_source_id).to_string(),
            schema.field_corpus => get_text(origin, schema.field_corpus).to_string(),
            schema.field_attributes => self.builder.build_attributes(doc).await,
            schema.field_updated_at => get_date(origin, schema.field_updated_at),
        };
        if let Some(failed_chunks) = get_number_optional(origin, schema.field_failed_chunks_count) {
            doc.add_u64(schema.field_failed_chunks_count, failed_chunks as u64);
        }

        doc
    }
}

pub struct Indexer {
    corpus: String,
    searcher: Searcher,
    writer: IndexWriter,
}

impl Indexer {
    pub fn new(corpus: &str) -> Self {
        let doc = IndexSchema::instance();
        let (_, index) = open_or_create_index(&doc.schema, &path::index_dir());
        let writer = index
            .writer(150_000_000)
            .expect("Failed to create index writer");
        let reader = index.reader().expect("Failed to create index reader");

        Self {
            corpus: corpus.to_owned(),
            searcher: reader.searcher(),
            writer,
        }
    }

    pub async fn add(&self, document: TantivyDocument) {
        self.writer
            .add_document(document)
            .expect("Failed to add document");
    }

    pub async fn get_doc(&self, id: &str) -> Result<TantivyDocument> {
        let schema = IndexSchema::instance();
        let query = schema.doc_query(&self.corpus, id);
        let docs = match self.searcher.search(&query, &TopDocs::with_limit(1)) {
            Ok(docs) => docs,
            Err(e) => {
                debug!("query tantivy error: {}", e);
                return Err(e.into());
            }
        };
        if docs.is_empty() {
            bail!("Document not found: {}", id);
        }

        self.searcher
            .doc(docs.first().unwrap().1)
            .map_err(|e| e.into())
    }

    // `get_doc_kind` returns the kind of a structured_doc, and `None` for a code.
    pub async fn get_doc_kind<'a>(&self, id: &str) -> Result<Option<String>> {
        let doc = self.get_doc(id).await?;
        let schema = IndexSchema::instance();
        Ok(get_json_text_optional(&doc, schema.field_attributes, KIND).map(|v| v.to_owned()))
    }

    /// Lists the latest document IDs based on the given source ID, key-value pairs, and datetime field.
    ///
    /// The IDs are sorted by the datetime field in descending order and filtered by the given constraints.
    pub async fn list_latest_ids(
        &self,
        source_id: &str,
        kvs: &Vec<(&str, &str)>,
        datetime_field: &str,
        offset: usize,
    ) -> Result<Vec<String>> {
        let schema = IndexSchema::instance();
        let query = schema.doc_with_attribute_field(&self.corpus, source_id, kvs);
        let docs = match self
            .searcher
            .search(&query, &TopDocs::with_limit(u16::MAX as usize))
        {
            Ok(docs) => docs,
            Err(e) => {
                debug!("query tantivy error: {}", e);
                return Err(e.into());
            }
        };
        if docs.is_empty() {
            bail!("No document found: {:?}", kvs);
        }

        let mut documents = Vec::new();
        for (_, doc_address) in docs {
            let doc: TantivyDocument = self.searcher.doc(doc_address)?;
            documents.push((
                get_text(&doc, schema.field_id).to_owned(),
                get_json_date_field(&doc, schema.field_attributes, datetime_field),
            ));
        }

        documents.sort_by(|a, b| b.1.cmp(&a.1));

        Ok(documents
            .iter()
            .skip(offset)
            .map(|(id, _)| id.to_owned())
            .collect())
    }

    pub async fn count_doc_by_attribute(
        &self,
        source_id: &str,
        kvs: &Vec<(&str, &str)>,
    ) -> Result<usize> {
        let schema = IndexSchema::instance();
        let query = schema.doc_with_attribute_field(&self.corpus, source_id, kvs);

        let count = self.searcher.search(&query, &tantivy::collector::Count)?;
        Ok(count)
    }

    pub fn delete(&self, id: &str) {
        let schema = IndexSchema::instance();
        let _ = self
            .writer
            .delete_query(Box::new(schema.doc_query_with_chunks(&self.corpus, id)));
    }

    pub fn delete_doc(&self, id: &str) {
        let schema = IndexSchema::instance();
        let _ = self
            .writer
            .delete_query(Box::new(schema.doc_query(&self.corpus, id)));
    }

    pub fn commit(mut self) {
        self.writer.commit().expect("Failed to commit changes");
        self.writer
            .wait_merging_threads()
            .expect("Failed to wait for merging threads");
    }

    // Check whether the document ID presents in the corpus.
    pub fn is_indexed(&self, id: &str) -> bool {
        let schema = IndexSchema::instance();
        let query = schema.doc_query(&self.corpus, id);
        let Ok(docs) = self.searcher.search(&query, &TopDocs::with_limit(1)) else {
            return false;
        };
        !docs.is_empty()
    }

    /// Iterates over all the document IDs in the corpus.
    pub fn iter_ids(&self) -> impl Stream<Item = (String, String)> + '_ {
        let schema = IndexSchema::instance();

        stream! {
            // Based on https://github.com/quickwit-oss/tantivy/blob/main/examples/iterating_docs_and_positions.rs
            for (segment_ordinal, segment_reader) in self.searcher.segment_readers().iter().enumerate() {
                let Ok(inverted_index) = segment_reader.inverted_index(schema.field_corpus) else {
                    continue;
                };

                let term_corpus = Term::from_field_text(schema.field_corpus, &self.corpus);
                let Ok(Some(mut postings)) = inverted_index.read_postings(&term_corpus, tantivy::schema::IndexRecordOption::Basic) else {
                    continue;
                };

                let mut doc_id = postings.doc();
                while doc_id != TERMINATED {
                    if !segment_reader.is_deleted(doc_id) {
                        let doc_address = DocAddress::new(segment_ordinal as u32, doc_id);
                        let doc: TantivyDocument = self.searcher.doc(doc_address).expect("Failed to read document");

                        // Skip chunks, as we only want to iterate over the main docs
                        if doc.get_first(schema.field_chunk_id).is_none() {
                            let id = get_text(&doc, schema.field_id);
                            let source = get_text(&doc, schema.field_source_id);
                            yield (source.to_owned(), id.to_owned());
                        }
                    }
                    doc_id = postings.advance();
                }
            }
        }
    }

    pub fn is_indexed_after(&self, id: &str, time: chrono::DateTime<chrono::Utc>) -> bool {
        let schema = IndexSchema::instance();
        let query = schema.doc_indexed_after(&self.corpus, id, time);
        let Ok(docs) = self.searcher.search(&query, &TopDocs::with_limit(1)) else {
            return false;
        };

        !docs.is_empty()
    }

    /// Check whether the document has failed chunks.
    ///
    /// failed chunks tracks the number of embedding indexing failed chunks for a document.
    pub fn has_failed_chunks(&self, id: &str) -> bool {
        let schema = IndexSchema::instance();
        let query = schema.doc_has_failed_chunks(&self.corpus, id);
        let Ok(docs) = self.searcher.search(&query, &TopDocs::with_limit(1)) else {
            return false;
        };

        !docs.is_empty()
    }

    // Check whether the document has attribute field.
    pub fn has_attribute_field(&self, id: &str, field: &str) -> bool {
        let schema = IndexSchema::instance();
        let query = schema.doc_has_attribute_field(&self.corpus, id, field);
        match self.searcher.search(&query, &TopDocs::with_limit(1)) {
            Ok(docs) => !docs.is_empty(),
            Err(e) => {
                debug!("query tantivy error: {}", e);
                false
            }
        }
    }
}

pub struct IndexGarbageCollector {
    searcher: Searcher,
    writer: IndexWriter,
}

impl IndexGarbageCollector {
    pub fn new() -> Self {
        let doc = IndexSchema::instance();
        let (_, index) = open_or_create_index(&doc.schema, &path::index_dir());
        let writer = index
            .writer(150_000_000)
            .expect("Failed to create index writer");
        let reader = index.reader().expect("Failed to create index reader");

        Self {
            searcher: reader.searcher(),
            writer,
        }
    }

    pub fn garbage_collect(&self, active_source_ids: &[String]) -> anyhow::Result<()> {
        let source_ids: HashSet<_> = active_source_ids.iter().collect();

        let count_aggregation: Aggregation = serde_json::from_value(json!({
            "terms": {
                "field": FIELD_SOURCE_ID,
            }
        }))
        .unwrap();

        let collector = AggregationCollector::from_aggs(
            vec![("count".to_owned(), count_aggregation)]
                .into_iter()
                .collect(),
            Default::default(),
        );

        let res = self.searcher.search(&AllQuery, &collector)?;
        let Some(AggregationResult::BucketResult(BucketResult::Terms { buckets, .. })) =
            res.0.get("count")
        else {
            bail!("Failed to get source_id count");
        };
        for source_id in buckets {
            let count = source_id.doc_count;
            let Key::Str(source_id) = &source_id.key else {
                warn!("Failed to get source_id key as string");
                continue;
            };

            if !source_ids.contains(source_id) {
                debug!("Deleting {} documents for source_id: {}", count, source_id,);
                self.delete_by_source_id(source_id);
            }
        }

        Ok(())
    }

    fn delete_by_source_id(&self, source_id: &str) {
        let schema = IndexSchema::instance();
        let _ = self
            .writer
            .delete_query(Box::new(schema.source_id_query(source_id)));
    }

    pub fn commit(mut self) {
        self.writer.commit().expect("Failed to commit changes");
        self.writer
            .wait_merging_threads()
            .expect("Failed to wait for merging threads");
    }
}

fn get_text(doc: &TantivyDocument, field: schema::Field) -> &str {
    doc.get_first(field).unwrap().as_str().unwrap()
}

fn get_date(doc: &TantivyDocument, field: schema::Field) -> tantivy::DateTime {
    doc.get_first(field).unwrap().as_datetime().unwrap()
}

fn get_number_optional(doc: &TantivyDocument, field: schema::Field) -> Option<i64> {
    doc.get_first(field)?.as_i64()
}

fn get_json_field<'a>(
    doc: &'a TantivyDocument,
    field: schema::Field,
    name: &str,
) -> CompactDocValue<'a> {
    doc.get_first(field)
        .unwrap()
        .as_object()
        .unwrap()
        .find(|(k, _)| *k == name)
        .unwrap()
        .1
}

fn get_json_date_field(doc: &TantivyDocument, field: schema::Field, name: &str) -> DateTime {
    get_json_field(doc, field, name).as_datetime().unwrap()
}

fn get_json_field_optional<'a>(
    doc: &'a TantivyDocument,
    field: schema::Field,
    name: &str,
) -> Option<CompactDocValue<'a>> {
    Some(
        doc.get_first(field)?
            .as_object()?
            .find(|(k, _)| *k == name)?
            .1,
    )
}

fn get_json_text_optional<'a>(
    doc: &'a TantivyDocument,
    field: schema::Field,
    name: &str,
) -> Option<&'a str> {
    get_json_field_optional(doc, field, name).map(|v| v.as_str().unwrap())
}
