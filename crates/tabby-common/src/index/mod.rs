pub mod code;
pub mod doc;

use std::borrow::Cow;

use lazy_static::lazy_static;
use tantivy::{
    query::{BooleanQuery, ConstScoreQuery, ExistsQuery, Occur, Query, TermQuery},
    schema::{
        Field, IndexRecordOption, JsonObjectOptions, Schema, TextFieldIndexing, FAST, INDEXED,
        STORED, STRING,
    },
    Term,
};

pub struct IndexSchema {
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

impl IndexSchema {
    pub fn instance() -> &'static Self {
        &INDEX_SCHEMA
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
    static ref INDEX_SCHEMA: IndexSchema = IndexSchema::new();
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
    embedding_dims: usize,
    embedding: impl Iterator<Item = &'a f32> + 'a,
) -> BooleanQuery {
    let schema = IndexSchema::instance();
    let iter = binarize_embedding(embedding).map(Cow::Owned);
    new_multiterms_const_query(schema.field_chunk_tokens, embedding_dims, iter)
}

fn new_multiterms_const_query<'a>(
    field: Field,
    embedding_dims: usize,
    terms: impl Iterator<Item = Cow<'a, str>> + 'a,
) -> BooleanQuery {
    let subqueries: Vec<Box<dyn Query>> = terms
        .map(|text| {
            let term = Term::from_field_text(field, text.as_ref());
            let term_query: Box<dyn Query> =
                Box::new(TermQuery::new(term, IndexRecordOption::Basic));

            let score = 1.0 / embedding_dims as f32;
            let boxed: Box<dyn Query> = Box::new(ConstScoreQuery::new(term_query, score));

            boxed
        })
        .collect();

    BooleanQuery::union(subqueries)
}

#[cfg(test)]
mod tests {

    use tantivy::{
        collector::TopDocs,
        query::Query,
        schema::{Schema, STRING},
        Index, IndexWriter, TantivyDocument,
    };

    use super::*;

    #[test]
    fn test_new_multiterms_const_query() -> anyhow::Result<()> {
        let mut schema_builder = Schema::builder();
        let field1 = schema_builder.add_text_field("field1", STRING);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer(15_000_000)?;

            // doc1
            let mut doc = TantivyDocument::new();
            doc.add_text(field1, "value1");
            doc.add_text(field1, "value2");
            doc.add_text(field1, "value3");
            index_writer.add_document(doc)?;

            // doc2
            let mut doc = TantivyDocument::new();
            doc.add_text(field1, "value2");
            doc.add_text(field1, "value4");
            index_writer.add_document(doc)?;

            index_writer.commit()?;
        }
        let reader = index.reader()?;
        let searcher = reader.searcher();

        {
            let query = new_multiterms_const_query(
                field1,
                4,
                vec!["value1", "value3"].into_iter().map(Cow::Borrowed),
            );

            let top_docs = searcher.search(&query, &TopDocs::with_limit(1))?;
            eprintln!("explain {:?}", query.explain(&searcher, top_docs[0].1)?);

            assert_eq!(top_docs.len(), 1, "Expected 1 document");
            assert_eq!(top_docs[0].0, 0.5);
        }

        {
            let query = new_multiterms_const_query(
                field1,
                4,
                vec!["value1", "value2", "value3"]
                    .into_iter()
                    .map(Cow::Borrowed),
            );

            let top_docs = searcher.search(&query, &TopDocs::with_limit(1))?;

            assert_eq!(top_docs.len(), 1, "Expected 1 document");
            assert_eq!(top_docs[0].0, 0.75);
        }

        Ok(())
    }
}
