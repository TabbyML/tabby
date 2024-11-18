use std::collections::HashSet;

use anyhow::bail;
use async_stream::stream;
use futures::{stream::BoxStream, Stream, StreamExt};
use serde_json::json;
use tabby_common::{
    index::{IndexSchema, FIELD_SOURCE_ID},
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
    schema::{self, Value},
    DocAddress, DocSet, IndexWriter, Searcher, TantivyDocument, Term, TERMINATED,
};
use tokio::task::JoinHandle;
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
    ) -> BoxStream<'a, JoinHandle<(Vec<String>, serde_json::Value)>>;
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

        let doc = doc! {
            schema.field_id => id,
            schema.field_source_id => source_id,
            schema.field_corpus => self.corpus,
            schema.field_attributes => self.builder.build_attributes(&document).await,
            schema.field_updated_at => updated_at,
        };

        let cloned_id = id.clone();
        let s = stream! {
            yield tokio::spawn(async move { Some(doc) });

            for await doc in self.build_chunks(cloned_id, source_id, updated_at, document).await {
                yield doc;
            }
        };

        (id, s)
    }

    async fn build_chunks(
        &self,
        id: String,
        source_id: String,
        updated_at: tantivy::DateTime,
        document: T,
    ) -> impl Stream<Item = JoinHandle<Option<TantivyDocument>>> + '_ {
        let kind = self.corpus;
        stream! {
            let schema = IndexSchema::instance();
            for await (chunk_id, task) in self.builder.build_chunk_attributes(&document).await.enumerate() {
                let id = id.clone();
                let source_id = source_id.clone();

                yield tokio::spawn(async move {
                    let Ok((tokens, chunk_attributes)) = task.await else {
                        return None;
                    };
                    if tokens.is_empty() {
                        return None;
                    }

                    let mut doc = doc! {
                        schema.field_id => id,
                        schema.field_source_id => source_id,
                        schema.field_corpus => kind,
                        schema.field_updated_at => updated_at,
                        schema.field_chunk_id => format!("{}-{}", id, chunk_id),
                        schema.field_chunk_attributes => chunk_attributes,
                    };

                    for token in tokens {
                        doc.add_text(schema.field_chunk_tokens, token);
                    }

                    Some(doc)
                });
            }
        }
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

    pub fn delete(&self, id: &str) {
        let schema = IndexSchema::instance();
        let _ = self
            .writer
            .delete_query(Box::new(schema.doc_query_with_chunks(&self.corpus, id)));
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
    pub fn iter_ids(&self) -> impl Stream<Item = String> + '_ {
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
                            yield id.to_owned();
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
