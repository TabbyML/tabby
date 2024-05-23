mod code;
use std::borrow::Cow;

pub use code::CodeSearchSchema;

mod doc;
pub use doc::{webcode, webdoc, DocSearchSchema};
use tantivy::{
    query::{BooleanQuery, ConstScoreQuery, Query, TermQuery},
    schema::{Field, IndexRecordOption},
    Term,
};

fn new_multiterms_const_query_with_path<'a>(
    field: Field,
    embedding_dims: usize,
    path: &str,
    terms: impl Iterator<Item = Cow<'a, str>> + 'a,
) -> BooleanQuery {
    let subqueries: Vec<Box<dyn Query>> = terms
        .map(|text| {
            let mut term = Term::from_field_json_path(field, path, false);
            term.append_type_and_str(text.as_ref());
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
    use serde_json::json;
    use tantivy::{
        collector::TopDocs,
        doc,
        query::Query,
        schema::{Schema, STRING},
        Index, IndexWriter,
    };

    use super::*;

    const PATH: &str = "attr";

    #[test]
    fn test_new_multiterms_const_query() -> anyhow::Result<()> {
        let mut schema_builder = Schema::builder();
        let field1 = schema_builder.add_json_field("field1", STRING);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer(15_000_000)?;

            // doc1
            let doc = doc!(
                field1 => json!({
                        PATH: vec!["value1", "value2", "value3"]
                })
            );
            index_writer.add_document(doc)?;

            // doc2
            let doc = doc!(
                field1 => json!({
                        PATH: vec!["value2", "value4"]
                })
            );
            index_writer.add_document(doc)?;

            index_writer.commit()?;
        }
        let reader = index.reader()?;
        let searcher = reader.searcher();

        {
            let query = new_multiterms_const_query_with_path(
                field1,
                4,
                PATH,
                vec!["value1", "value3"].into_iter().map(Cow::Borrowed),
            );

            let top_docs = searcher.search(&query, &TopDocs::with_limit(1))?;
            eprintln!("explain {:?}", query.explain(&searcher, top_docs[0].1)?);

            assert_eq!(top_docs.len(), 1, "Expected 1 document");
            assert_eq!(top_docs[0].0, 0.5);
        }

        {
            let query = new_multiterms_const_query_with_path(
                field1,
                4,
                PATH,
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
