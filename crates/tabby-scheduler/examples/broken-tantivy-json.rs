use serde_json::json;
use tantivy::{collector::TopDocs, doc, query::TermQuery, schema::*, Index, IndexWriter};

const JSON: &str = "json";

// # Document from json
//
// For convenience, `Document` can be parsed directly from json.
fn main() -> tantivy::Result<()> {
    // Let's first define a schema and an index.
    // Check out the basic example if this is confusing to you.
    //
    // first we need to define a schema ...
    let mut schema_builder = Schema::builder();

    let field_json = schema_builder.add_json_field(
        JSON,
        JsonObjectOptions::default()
            .set_stored()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("raw")
                    .set_fieldnorms(true)
                    .set_index_option(tantivy::schema::IndexRecordOption::Basic),
            ),
    );

    let schema = schema_builder.build();

    let index = Index::create_in_ram(schema.clone());

    // Multi-valued field are allowed, they are
    // expressed in JSON by an array.
    // The following document has two titles.
    let frankenstein_json = json! { {
       "title": "Frankenstein",
       "year": 1818
    }};
    let _frankenstein_doc = doc!(
       field_json => frankenstein_json
    );

    let mut writer: IndexWriter<TantivyDocument> = index.writer(15000000)?;
    writer.add_document(_frankenstein_doc)?;
    writer.commit()?;
    drop(writer);

    let mut writer: IndexWriter<TantivyDocument> = index.writer(15000000)?;

    let mut term = Term::from_field_json_path(field_json, "title", false);
    term.append_type_and_str("Frankenstein");
    writer.delete_term(term.clone());
    writer.commit()?;
    writer.wait_merging_threads()?;

    let reader = index.reader()?;

    let searcher = reader.searcher();
    let search_results = searcher.search(
        &TermQuery::new(term, IndexRecordOption::Basic),
        &TopDocs::with_limit(1),
    )?;
    dbg!(search_results);

    for segment in reader.searcher().segment_readers() {
        let Ok(field) = segment.inverted_index(field_json) else {
            continue;
        };
        let terms = field.terms();
        let stream = terms.stream()?;
        let info = stream.value();
        println!("{info:#?}");
    }

    // writer.delete_term()

    // Note that the schema is saved in your index directory.
    //
    // As a result, Indexes are aware of their schema, and you can use this feature
    // just by opening an existing `Index`, and calling `index.schema()..parse_document(json)`.
    Ok(())
}
