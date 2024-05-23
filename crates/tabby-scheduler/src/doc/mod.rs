mod web;

use std::sync::Arc;

use futures::{Stream, StreamExt};
use tabby_common::{index::DocSearchSchema, path};
use tabby_inference::Embedding;
use tantivy::{doc, IndexWriter, TantivyDocument, Term};

use self::web::WebBuilder;
use crate::{tantivy_utils::open_or_create_index, DocumentBuilder};

pub struct SourceDocument {
    pub id: String,
    pub title: String,
    pub link: String,
    pub body: String,
}

pub struct DocIndex<T> {
    builder: Box<dyn DocumentBuilder<T>>,
    writer: IndexWriter,
}

impl<T> DocIndex<T> {
    pub fn new(builder: impl DocumentBuilder<T> + 'static) -> Self {
        let doc = DocSearchSchema::instance();
        let (_, index) = open_or_create_index(&doc.schema, &path::doc_index_dir());
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

pub fn create_web_index(embedding: Arc<dyn Embedding>) -> DocIndex<SourceDocument> {
    let builder = WebBuilder::new(embedding);
    DocIndex::new(builder)
}

#[cfg(test)]
mod tests {
    use core::panic;

    use async_trait::async_trait;
    use tabby_common::index::webdoc;
    use tantivy::schema::{
        document::{CompactDocValue, ReferenceValue},
        Field, Value,
    };

    use super::*;

    struct DummyEmbedding;

    #[async_trait]
    impl Embedding for DummyEmbedding {
        async fn embed(&self, prompt: &str) -> anyhow::Result<Vec<f32>> {
            Ok(vec![0.0; prompt.len()])
        }
    }

    fn new_test_index() -> DocIndex<SourceDocument> {
        create_web_index(Arc::new(DummyEmbedding))
    }

    fn is_empty(doc: &TantivyDocument, field: tantivy::schema::Field) -> bool {
        doc.get_first(field).is_none()
    }

    fn get_text(doc: &TantivyDocument, field: tantivy::schema::Field) -> String {
        doc.get_first(field).unwrap().as_str().unwrap().to_string()
    }

    fn get_json_array_field<'a>(
        doc: &'a TantivyDocument,
        field: Field,
        name: &str,
    ) -> impl Iterator<Item = CompactDocValue<'a>> {
        let ReferenceValue::Object(object) = doc.get_first(field).unwrap() else {
            panic!("Field {:?} is not an array", field);
        };
        let pair = object.into_iter().find(|(key, _)| *key == name).unwrap();
        pair.1.as_array().unwrap()
    }

    #[tokio::test]
    async fn test_iter_docs() {
        let index = new_test_index();
        let document = SourceDocument {
            id: "test".to_string(),
            title: "Test".to_string(),
            link: "https://example.com".to_string(),
            body: "Hello, world!".to_string(),
        };
        let docs = index.iter_docs(document).await.collect::<Vec<_>>().await;
        assert_eq!(2, docs.len());

        let schema = DocSearchSchema::instance();

        // Check document
        assert_eq!("test", get_text(&docs[0], schema.field_id));
        assert!(is_empty(&docs[0], schema.field_chunk_id));
        assert!(is_empty(&docs[0], schema.field_chunk_attributes));

        // Check chunks.
        assert_eq!("test", get_text(&docs[1], schema.field_id));
        assert!(is_empty(&docs[1], schema.field_attributes));

        assert_eq!("test-0", get_text(&docs[1], schema.field_chunk_id));
        assert_eq!(
            "embedding_zero_0",
            get_json_array_field(
                &docs[1],
                schema.field_chunk_attributes,
                webdoc::fields::CHUNK_EMBEDDING
            )
            .next()
            .unwrap()
            .as_str()
            .unwrap()
        );
    }
}
