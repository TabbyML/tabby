use std::{collections::HashSet, sync::Arc};

use tabby_common::{index::DocSearchSchema, path};
use tabby_inference::Embedding;
use tantivy::{doc, Index, IndexWriter, Term};
use text_splitter::{Characters, TextSplitter};
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
    splitter: TextSplitter<Characters>,
}

const CHUNK_SIZE: usize = 2048;

fn make_embedding_token(i: usize) -> String {
    format!("embedding_{i}")
}

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
            splitter: TextSplitter::default().with_trim_chunks(true),
        }
    }

    pub async fn add(&mut self, document: SourceDocument) {
        // Delete the document if it already exists
        self.writer
            .delete_term(Term::from_field_text(self.doc.field_id, &document.id));

        let Some(embedding_tokens) = self.compute_embedding_tokens(&document.body).await else {
            warn!(
                "Failed to compute embedding tokens for document '{}'",
                document.id
            );
            return;
        };

        let mut doc = doc! {
            self.doc.field_id => document.id,
            self.doc.field_title => document.title,
            self.doc.field_link => document.link,
            self.doc.field_body => document.body,
        };

        for token in embedding_tokens {
            doc.add_text(self.doc.field_embedding_token, token);
        }

        // Add the document
        self.writer
            .add_document(doc)
            .expect("Failed to add document");
    }

    /// This function splits the document into chunks and computes the embedding for each chunk. It then converts the embeddings
    /// into binarized tokens by thresholding on zero.
    ///
    /// The current implementation deduplicates tokens at the document level, but this may require further consideration in the future.
    async fn compute_embedding_tokens(&self, content: &str) -> Option<Vec<String>> {
        let mut tokens = HashSet::new();
        for chunk in self.splitter.chunks(content, CHUNK_SIZE) {
            let embedding = match self.embedding.embed(chunk).await {
                Ok(embedding) => embedding,
                Err(e) => {
                    warn!("Failed to embed document: {}", e);
                    return None;
                }
            };

            for (i, value) in embedding.iter().enumerate() {
                if *value > 0.0 {
                    tokens.insert(make_embedding_token(i));
                }
            }
        }

        Some(tokens.into_iter().collect())
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
