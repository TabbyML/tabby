use std::borrow::Cow;

use tantivy::{
    query::{BooleanQuery, ConstScoreQuery, Query, TermQuery},
    schema::{Field, IndexRecordOption},
    Term,
};

use super::DocSearchSchema;

pub mod fields {
    pub const TITLE: &str = "title";
    pub const LINK: &str = "link";
    pub const CHUNK_TEXT: &str = "chunk_text";
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
    let schema = DocSearchSchema::instance();
    let iter = binarize_embedding(embedding).map(Cow::Owned);
    new_multiterms_const_query(schema.field_chunk_tokens, embedding_dims, iter)
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
