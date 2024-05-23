use futures::{stream::BoxStream, Stream, StreamExt};
use tabby_common::{index::DocSearchSchema, path};
use tantivy::{doc, IndexWriter, TantivyDocument, Term};

use crate::tantivy_utils::open_or_create_index;

#[async_trait::async_trait]
pub trait DocumentBuilder<T>: Send + Sync {
    async fn build_id(&self, document: &T) -> String;
    async fn build_attributes(&self, document: &T) -> serde_json::Value;
    async fn build_chunk_attributes(&self, document: &T) -> BoxStream<serde_json::Value>;
}

pub struct DocIndex<T> {
    builder: Box<dyn DocumentBuilder<T>>,
    writer: IndexWriter,
}

impl<T> DocIndex<T> {
    pub fn new(builder: impl DocumentBuilder<T> + 'static) -> Self {
        let doc = DocSearchSchema::instance();
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
        let schema = DocSearchSchema::instance();
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
        let schema = DocSearchSchema::instance();
        self.builder
            .build_chunk_attributes(&document)
            .await
            .enumerate()
            .map(move |(chunk_id, chunk_attributes)| {
                doc! {
                    schema.field_id => id,
                    schema.field_updated_at => updated_at,
                    schema.field_chunk_id => format!("{}-{}", id, chunk_id),
                    schema.field_chunk_attributes => chunk_attributes,
                }
            })
    }

    pub fn delete(&self, id: &str) {
        self.writer.delete_term(Term::from_field_text(
            DocSearchSchema::instance().field_id,
            id,
        ));
    }

    pub fn commit(mut self) {
        self.writer.commit().expect("Failed to commit changes");
        self.writer
            .wait_merging_threads()
            .expect("Failed to wait for merging threads");
    }
}
