use std::path::Path;

use clap::Args;
use tabby_common::index::IndexSchema;
use tantivy::{DocAddress, DocSet, Document, Index, TantivyDocument, Term, TERMINATED};

#[derive(Args)]
pub struct HeadArgs {
    /// Number of documents to display
    #[clap(short, long, default_value = "1")]
    num_docs: usize,

    #[clap(short, long, default_value = "code")]
    corpus: String,
}

pub fn run_head_cli(index_path: &Path, args: &HeadArgs) -> anyhow::Result<()> {
    let index = Index::open_in_dir(index_path)?;

    let searcher = index.reader()?.searcher();
    let schema = IndexSchema::instance();

    let mut count = 0;
    'outer: for (segment_ordinal, segment_reader) in searcher.segment_readers().iter().enumerate() {
        let Ok(inverted_index) = segment_reader.inverted_index(schema.field_corpus) else {
            continue;
        };

        let term_corpus = Term::from_field_text(schema.field_corpus, &args.corpus);
        let Ok(Some(mut postings)) =
            inverted_index.read_postings(&term_corpus, tantivy::schema::IndexRecordOption::Basic)
        else {
            continue;
        };

        let mut doc_id = postings.doc();
        while doc_id != TERMINATED {
            if !segment_reader.is_deleted(doc_id) {
                let doc_address = DocAddress::new(segment_ordinal as u32, doc_id);
                let doc: TantivyDocument =
                    searcher.doc(doc_address).expect("Failed to read document");

                let json_value = to_json_value(doc, &schema.schema);

                println!("{json_value}");

                count += 1;
                if count >= args.num_docs {
                    break 'outer;
                }
            }
            doc_id = postings.advance();
        }
    }

    Ok(())
}

fn to_json_value(doc: TantivyDocument, schema: &tantivy::schema::Schema) -> serde_json::Value {
    let json = doc.to_json(schema);
    let mut doc: serde_json::Value = serde_json::from_str(&json).expect("Failed to parse JSON");

    for (_, value) in doc.as_object_mut().expect("Expected object").iter_mut() {
        if let Some(array) = value.as_array_mut() {
            if array.len() == 1 {
                *value = array[0].clone();
            }
        }
    }

    doc
}
