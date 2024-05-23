use std::borrow::Cow;

use lazy_static::lazy_static;
use tantivy::{
    query::{BooleanQuery, ExistsQuery, Occur, TermQuery}, schema::{Field, JsonObjectOptions, Schema, TextFieldIndexing, FAST, INDEXED, STORED, STRING}, tokenizer::TokenizerManager, Term
};

use super::new_multiterms_const_query;

pub mod webdoc {
    pub mod fields {
        pub const TITLE: &str = "title";
        pub const LINK: &str = "link";
        pub const CHUNK_TEXT: &str = "chunk_text";
    }
}

pub mod webcode {
    pub mod fields {
        pub const CHUNK_GIT_URL: &str = "chunk_git_url";
        pub const CHUNK_FILEPATH: &str = "chunk_filepath";
        pub const CHUNK_LANGUAGE: &str = "chunk_language";
        pub const CHUNK_BODY: &str = "chunk_body";
        pub const CHUNK_START_LINE: &str = "chunk_start_line";
    }
}

pub struct DocSearchSchema {
    pub schema: Schema,

    // === Fields for both document and chunk ===
    pub field_id: Field,
    pub field_updated_at: Field,

    // === Fields for document ===
    pub field_attributes: Field,

    // === Fields for chunk ===
    pub field_chunk_id: Field,
    pub field_chunk_attributes: Field,

    pub field_chunk_tokens: Field,
}

const FIELD_CHUNK_ID: &str = "chunk_id";

impl DocSearchSchema {
    pub fn instance() -> &'static Self {
        &DOC_SEARCH_SCHEMA
    }

    fn new() -> Self {
        let mut builder = Schema::builder();

        let field_id = builder.add_text_field("id", STRING | STORED);
        let field_updated_at = builder.add_date_field("updated_at", INDEXED);
        let field_attributes = builder.add_text_field("attributes", STORED);

        let field_chunk_id = builder.add_text_field(FIELD_CHUNK_ID, STRING | FAST | STORED);
        let field_chunk_attributes = builder.add_json_field(
            "chunk_attributes",
            JsonObjectOptions::default()
                .set_stored()
                .set_indexing_options(
                    TextFieldIndexing::default()
                        .set_tokenizer("raw")
                        .set_fieldnorms(true)
                        .set_index_option(tantivy::schema::IndexRecordOption::Basic)
                        .set_fieldnorms(true),
                ),
        );

        let field_chunk_tokens = builder.add_text_field("chunk_tokens", STRING);
        let schema = builder.build();

        Self {
            schema,
            field_id,
            field_updated_at,
            field_attributes,

            field_chunk_id,
            field_chunk_attributes,
            field_chunk_tokens,
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
        embedding_dims: usize,
        embedding: impl Iterator<Item = &'a f32> + 'a,
    ) -> BooleanQuery {
        let iter = DocSearchSchema::binarize_embedding(embedding).map(Cow::Owned);

        new_multiterms_const_query(
            self.field_chunk_tokens,
            embedding_dims,
            iter,
        )
    }

    /// Build a query to find the document with the given `doc_id`.
    pub fn doc_query(&self, doc_id: &str) -> BooleanQuery {
        let doc_id_query = TermQuery::new(
            Term::from_field_text(self.field_id, doc_id),
            tantivy::schema::IndexRecordOption::Basic,
        );

        BooleanQuery::new(vec![
            // Must match the doc id
            (Occur::Must, Box::new(doc_id_query)),
            // Exclude chunk documents
            (
                Occur::MustNot,
                Box::new(ExistsQuery::new_exists_query(FIELD_CHUNK_ID.into())),
            ),
        ])
    }
}

lazy_static! {
    static ref DOC_SEARCH_SCHEMA: DocSearchSchema = DocSearchSchema::new();
}
