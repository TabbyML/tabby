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

        // Add the chunks
        self.iter_chunks(document.id.clone(), document.body.clone())
            .await
            .for_each(|doc| async {
                self.writer.add_document(doc).expect("Failed to add chunk");
            })
            .await;

        // Add the document
        let doc = doc! {
            self.doc.field_id => document.id,
            self.doc.field_title => document.title,
            self.doc.field_link => document.link,
            self.doc.field_body => document.body,
        };

        // Add the document
        self.writer
            .add_document(doc)
            .expect("Failed to add document");
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

        let field_chunk_id = self.doc.field_chunk_id;
        let field_chunk_embedding_token = self.doc.field_chunk_embedding_token;
        stream! {
            for (chunk_id, chunk) in splitter.chunks(&content, CHUNK_SIZE).enumerate() {
                let mut doc = doc! {
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
