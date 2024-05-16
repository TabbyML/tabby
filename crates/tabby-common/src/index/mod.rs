mod code;
use std::borrow::Cow;

pub use code::{register_tokenizers, CodeSearchSchema};

mod doc;
pub use doc::DocSearchSchema;
use tantivy::{
    query::{BooleanQuery, ConstScoreQuery, Query, TermQuery},
    schema::{Field, IndexRecordOption},
    Term,
};

fn new_multiterms_const_query<'a>(
    field: Field,
    terms: impl Iterator<Item = Cow<'a, str>> + 'a,
) -> BooleanQuery {
    let subqueries: Vec<Box<dyn Query>> = terms
        .map(|term| {
            let term_query: Box<dyn Query> = Box::new(TermQuery::new(
                Term::from_field_text(field, term.as_ref()),
                IndexRecordOption::Basic,
            ));

            let boxed: Box<dyn Query> = Box::new(ConstScoreQuery::new(term_query, 1.0));

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
                vec!["value1", "value3"].into_iter().map(Cow::Borrowed),
            );

            let top_docs = searcher.search(&query, &TopDocs::with_limit(1))?;
            eprintln!("explain {:?}", query.explain(&searcher, top_docs[0].1)?);

            assert_eq!(top_docs.len(), 1, "Expected 1 document");
            assert_eq!(top_docs[0].0, 2.0);
        }

        {
            let query = new_multiterms_const_query(
                field1,
                vec!["value1", "value2", "value3"]
                    .into_iter()
                    .map(Cow::Borrowed),
            );

            let top_docs = searcher.search(&query, &TopDocs::with_limit(1))?;

            assert_eq!(top_docs.len(), 1, "Expected 1 document");
            assert_eq!(top_docs[0].0, 3.0);
        }

        Ok(())
    }
}
