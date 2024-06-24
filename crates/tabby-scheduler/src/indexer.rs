use async_stream::stream;
use futures::{stream::BoxStream, Stream, StreamExt};
use tabby_common::{index::IndexSchema, path};
use tantivy::{
    collector::TopDocs,
    doc,
    query::TermQuery,
    schema::{self, IndexRecordOption, Value},
    DocAddress, DocSet, IndexWriter, Searcher, TantivyDocument, Term, TERMINATED,
};
use tracing::{debug, info};

use crate::tantivy_utils::open_or_create_index;

#[async_trait::async_trait]
pub trait IndexAttributeBuilder<T>: Send + Sync {
    async fn build_id(&self, document: &T) -> String;
    async fn build_attributes(&self, document: &T) -> serde_json::Value;
    async fn build_chunk_attributes(
        &self,
        document: &T,
    ) -> BoxStream<(Vec<String>, serde_json::Value)>;
}

pub struct Indexer<T> {
    kind: &'static str,
    builder: Box<dyn IndexAttributeBuilder<T>>,
    searcher: Searcher,
    writer: IndexWriter,
    pub recreated: bool,
}

impl<T: Send + 'static> Indexer<T> {
    pub fn new(kind: &'static str, builder: impl IndexAttributeBuilder<T> + 'static) -> Self {
        let doc = IndexSchema::instance();
        let (recreated, index) = open_or_create_index(&doc.schema, &path::index_dir());
        let writer = index
            .writer(150_000_000)
            .expect("Failed to create index writer");
        let reader = index.reader().expect("Failed to create index reader");

        Self {
            kind,
            builder: Box::new(builder),
            searcher: reader.searcher(),
            writer,
            recreated,
        }
    }

    pub async fn add(&self, document: T) {
        self.build_doc(document)
            .await
            .for_each(|doc| {
                self.writer
                    .add_document(doc)
                    .expect("Failed to add document");
                async {}
            })
            .await;
    }

    async fn build_doc(&self, document: T) -> impl Stream<Item = TantivyDocument> + '_ {
        let schema = IndexSchema::instance();
        let id = self.format_id(&self.builder.build_id(&document).await);

        // Delete the document if it already exists
        self.writer
            .delete_term(Term::from_field_text(schema.field_id, &id));

        let now = tantivy::time::OffsetDateTime::now_utc();
        let updated_at = tantivy::DateTime::from_utc(now);

        let doc = doc! {
            schema.field_id => id,
            schema.field_corpus => self.kind,
            schema.field_attributes => self.builder.build_attributes(&document).await,
            schema.field_updated_at => updated_at,
        };

        futures::stream::once(async { doc })
            .chain(self.build_chunks(id, updated_at, document).await)
    }

    async fn build_chunks(
        &self,
        id: String,
        updated_at: tantivy::DateTime,
        document: T,
    ) -> impl Stream<Item = TantivyDocument> + '_ {
        let schema = IndexSchema::instance();
        self.builder
            .build_chunk_attributes(&document)
            .await
            .enumerate()
            .map(move |(chunk_id, (tokens, chunk_attributes))| {
                let mut doc = doc! {
                    schema.field_id => id,
                    schema.field_corpus => self.kind,
                    schema.field_updated_at => updated_at,
                    schema.field_chunk_id => format!("{}-{}", id, chunk_id),
                    schema.field_chunk_attributes => chunk_attributes,
                };

                for token in tokens {
                    doc.add_text(schema.field_chunk_tokens, token);
                }

                doc
            })
    }

    pub fn delete(&self, id: &str) {
        self.writer.delete_term(Term::from_field_text(
            IndexSchema::instance().field_id,
            &self.format_id(id),
        ));
    }

    fn format_id(&self, id: &str) -> String {
        format!("{}:{}", self.kind, id)
    }

    pub fn commit(mut self) {
        debug!("Committing changes to index...");
        self.writer.commit().expect("Failed to commit changes");
        self.writer
            .wait_merging_threads()
            .expect("Failed to wait for merging threads");
    }

    // Check whether the document ID presents in the corpus.
    pub fn is_indexed(&self, id: &str) -> bool {
        let schema = IndexSchema::instance();
        let query = TermQuery::new(
            Term::from_field_text(schema.field_id, &self.format_id(id)),
            IndexRecordOption::Basic,
        );
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

                let term_corpus = Term::from_field_text(schema.field_corpus, self.kind);
                let Ok(Some(mut postings)) = inverted_index.read_postings(&term_corpus, tantivy::schema::IndexRecordOption::Basic) else {
                    continue;
                };

                let prefix_to_strip = format!("{}:", self.kind);
                let mut doc_id = postings.doc();
                while doc_id != TERMINATED {
                    if !segment_reader.is_deleted(doc_id) {
                        let doc_address = DocAddress::new(segment_ordinal as u32, doc_id);
                        let doc: TantivyDocument = self.searcher.doc(doc_address).expect("Failed to read document");

                        // Skip chunks, as we only want to iterate over the main docs
                        if doc.get_first(schema.field_chunk_id).is_none() {
                            if let Some(id) = get_text(&doc, schema.field_id).strip_prefix(&prefix_to_strip) {
                                yield id.to_owned();
                            }
                        }
                    }
                    doc_id = postings.advance();
                }
            }
        }
    }
}

fn get_text(doc: &TantivyDocument, field: schema::Field) -> &str {
    doc.get_first(field).unwrap().as_str().unwrap()
}
