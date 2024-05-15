use std::sync::Arc;

use async_stream::stream;
use futures::{Stream, StreamExt};
use tabby_common::{index::DocSearchSchema, path};
use tabby_inference::Embedding;
use tantivy::{doc, IndexWriter, TantivyDocument, Term};
use text_splitter::TextSplitter;
use tracing::warn;

use crate::tantivy_utils::open_or_create_index;

pub struct SourceDocument {
    pub id: String,
    pub title: String,
    pub link: String,
    pub body: String,
}

pub struct DocIndex {
    embedding: Arc<dyn Embedding>,
    doc: DocSearchSchema,
    writer: IndexWriter,
}

const CHUNK_SIZE: usize = 2048;

impl DocIndex {
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        let doc = DocSearchSchema::default();
        let index = open_or_create_index(&doc.schema, &path::doc_index_dir());
        let writer = index
            .writer(150_000_000)
            .expect("Failed to create index writer");

        Self {
            embedding,
            doc,
            writer,
        }
    }

    pub async fn add(&mut self, document: SourceDocument) {
        // Delete the document if it already exists
        self.writer
            .delete_term(Term::from_field_text(self.doc.field_id, &document.id));

        self.iter_docs(document)
            .await
            .for_each(|doc| async {
                self.writer
                    .add_document(doc)
                    .expect("Failed to add document");
            })
            .await;
    }

    async fn iter_docs(&self, document: SourceDocument) -> impl Stream<Item = TantivyDocument> {
        let id = document.id.clone();
        let content = document.body.clone();

        let doc = doc! {
            self.doc.field_id => document.id,
            self.doc.field_title => document.title,
            self.doc.field_link => document.link,
            self.doc.field_body => document.body,
        };

        futures::stream::once(async { doc }).chain(self.iter_chunks(id, content).await)
    }

    /// This function splits the document into chunks and computes the embedding for each chunk. It then converts the embeddings
    /// into binarized tokens by thresholding on zero.
    async fn iter_chunks(
        &self,
        id: String,
        content: String,
    ) -> impl Stream<Item = TantivyDocument> {
        let splitter = TextSplitter::default().with_trim_chunks(true);
        let embedding = self.embedding.clone();

        let field_id = self.doc.field_id;
        let field_chunk_id = self.doc.field_chunk_id;
        let field_chunk_embedding_token = self.doc.field_chunk_embedding_token;
        stream! {
            for (chunk_id, chunk) in splitter.chunks(&content, CHUNK_SIZE).enumerate() {
                let mut doc = doc! {
                    field_id => id.clone(),
                    field_chunk_id => chunk_id.to_string()
                };

                let Ok(embedding) = embedding.embed(chunk).await else {
                    warn!("Failed to embed chunk {} of document '{}'", chunk_id, id);
                    continue;
                };

                for token in DocSearchSchema::binarize_embedding(embedding.iter()) {
                    doc.add_text(field_chunk_embedding_token, token);
                }

                yield doc;
            }
        }
    }

    pub fn delete(&mut self, id: &str) {
        self.writer
            .delete_term(Term::from_field_text(self.doc.field_id, id));
    }

    pub fn commit(mut self) {
        self.writer.commit().expect("Failed to commit changes");
        self.writer
            .wait_merging_threads()
            .expect("Failed to wait for merging threads");
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use tantivy::schema::Value;

    use super::*;

    struct DummyEmbedding;

    #[async_trait]
    impl Embedding for DummyEmbedding {
        async fn embed(&self, prompt: &str) -> anyhow::Result<Vec<f32>> {
            Ok(vec![0.0; prompt.len()])
        }
    }

    fn new_test_index() -> DocIndex {
        DocIndex::new(Arc::new(DummyEmbedding))
    }

    fn is_empty(doc: &TantivyDocument, field: tantivy::schema::Field) -> bool {
        doc.get_first(field).is_none()
    }

    fn get_text(doc: &TantivyDocument, field: tantivy::schema::Field) -> String {
        doc.get_first(field).unwrap().as_str().unwrap().to_string()
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

        // Check document
        assert_eq!("test", get_text(&docs[0], index.doc.field_id));
        assert!(is_empty(&docs[0], index.doc.field_chunk_id));
        assert!(is_empty(&docs[0], index.doc.field_chunk_embedding_token));

        // Check chunks.
        assert_eq!("test", get_text(&docs[1], index.doc.field_id));
        assert!(is_empty(&docs[1], index.doc.field_title));
        assert!(is_empty(&docs[1], index.doc.field_link));
        assert!(is_empty(&docs[1], index.doc.field_body));

        assert_eq!("0", get_text(&docs[1], index.doc.field_chunk_id));
        assert_eq!(
            "embedding_zero_0",
            get_text(&docs[1], index.doc.field_chunk_embedding_token)
        );
    }
}
