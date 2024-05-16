use lazy_static::lazy_static;
use tantivy::{
    query::{TermQuery, TermSetQuery},
    schema::{Field, Schema, STORED, STRING},
    Term,
};

pub struct DocSearchSchema {
    pub schema: Schema,

    // === Fields for document ===
    pub field_id: Field,
    pub field_title: Field,
    pub field_link: Field,

    // === Fields for chunk ===
    pub field_chunk_id: Field,
    pub field_chunk_text: Field,
    // Binarized embedding tokens with the following mapping:
    // * [-1, 0] -> 0
    // * (0, 1] -> 1
    pub field_chunk_embedding_token: Field,
}

impl DocSearchSchema {
    pub fn instance() -> &'static Self {
        &DOC_SEARCH_SCHEMA
    }

    fn new() -> Self {
        let mut builder = Schema::builder();

        let field_id = builder.add_text_field("id", STRING | STORED);
        let field_title = builder.add_text_field("title", STORED);
        let field_link = builder.add_text_field("link", STORED);

        let field_chunk_id = builder.add_text_field("chunk_id", STRING | STORED);
        let field_chunk_text = builder.add_text_field("chunk_text", STORED);
        let field_chunk_embedding_token = builder.add_text_field("chunk_embedding_token", STRING);

        let schema = builder.build();

        Self {
            schema,
            field_id,
            field_title,
            field_link,

            field_chunk_id,
            field_chunk_text,
            field_chunk_embedding_token,
        }
    }

    pub fn binarize_embedding<'a>(
        embedding: impl Iterator<Item = &'a f32> + 'a,
    ) -> impl Iterator<Item = String> + 'a {
        embedding.enumerate().map(|(i, value)| {
            if *value <= 0.0 {
                format!("embedding_zero_{}", i)
            } else {
                format!("embedding_one_{}", i)
            }
        })
    }

    pub fn embedding_tokens_query<'a>(
        &self,
        embedding: impl Iterator<Item = &'a f32> + 'a,
    ) -> TermSetQuery {
        let embedding_tokens = DocSearchSchema::binarize_embedding(embedding)
            .map(|x| Term::from_field_text(self.field_chunk_embedding_token, &x));
        TermSetQuery::new(embedding_tokens)
    }

    pub fn doc_query(&self, doc_id: &str) -> TermQuery {
        TermQuery::new(
            Term::from_field_text(self.field_id, doc_id),
            tantivy::schema::IndexRecordOption::Basic,
        )
    }
}

lazy_static! {
    static ref DOC_SEARCH_SCHEMA: DocSearchSchema = DocSearchSchema::new();
}
