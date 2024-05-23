use futures::{stream::BoxStream, Stream, StreamExt};
use tabby_common::{index::IndexSchema, path};
use tantivy::{doc, IndexWriter, TantivyDocument, Term};

use crate::tantivy_utils::open_or_create_index;

#[async_trait::async_trait]
pub trait IndexAttributeBuilder<T>: Send + Sync {
    fn format_id(&self, id: &str) -> String;
    async fn build_id(&self, document: &T) -> String;
    async fn build_attributes(&self, document: &T) -> serde_json::Value;
    async fn build_chunk_attributes(
        &self,
        document: &T,
    ) -> BoxStream<(Vec<String>, serde_json::Value)>;
}

pub struct Indexer<T> {
    builder: Box<dyn IndexAttributeBuilder<T>>,
    writer: IndexWriter,
}

impl<T> Indexer<T> {
    pub fn new(builder: impl IndexAttributeBuilder<T> + 'static) -> Self {
        let doc = IndexSchema::instance();
        let (_, index) = open_or_create_index(&doc.schema, &path::index_dir());
        let writer = index
            .writer(150_000_000)
            .expect("Failed to create index writer");

        Self {
            builder: Box::new(builder),
            writer,
        }
    }

    pub async fn add(&self, document: T) {
        self.iter_docs(document)
            .await
            .for_each(|doc| {
                self.writer
                    .add_document(doc)
                    .expect("Failed to add document");
                async {}
            })
            .await;
    }

    async fn iter_docs(&self, document: T) -> impl Stream<Item = TantivyDocument> + '_ {
        let schema = IndexSchema::instance();
        let id = self.builder.build_id(&document).await;

        // Delete the document if it already exists
        self.writer
            .delete_term(Term::from_field_text(schema.field_id, &id));

        let now = tantivy::time::OffsetDateTime::now_utc();
        let updated_at = tantivy::DateTime::from_utc(now);

        let doc = doc! {
            schema.field_id => id,
            schema.field_attributes => self.builder.build_attributes(&document).await,
            schema.field_updated_at => updated_at,
        };

        futures::stream::once(async { doc }).chain(self.iter_chunks(id, updated_at, document).await)
    }

    async fn iter_chunks(
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
            &self.builder.format_id(id),
        ));
    }

    pub fn commit(mut self) {
        self.writer.commit().expect("Failed to commit changes");
        self.writer
            .wait_merging_threads()
            .expect("Failed to wait for merging threads");
    }
}
