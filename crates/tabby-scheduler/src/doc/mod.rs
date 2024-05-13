use tabby_common::{index::DocSearchSchema, path};
use tantivy::{doc, Index, IndexWriter, Term};

use crate::tantivy_utils::open_or_create_index;

struct Document {
    pub id: String,
    pub title: String,
    pub link: String,
    pub snippet: String,
}

struct DocIndex {
    doc: DocSearchSchema,
    index: Index,
    writer: IndexWriter,
}

impl DocIndex {
    pub fn new() -> Self {
        let doc = DocSearchSchema::default();
        let index = open_or_create_index(&doc.schema, &path::doc_index_dir());
        let writer = index
            .writer(150_000_000)
            .expect("Failed to create index writer");

        Self { doc, index, writer }
    }

    pub fn add(&mut self, document: Document) {
        // Delete the document if it already exists
        self.writer
            .delete_term(Term::from_field_text(self.doc.field_id, &document.id));

        // Add the document
        self.writer
            .add_document(doc! {
                self.doc.field_id => document.id,
                // FIXME: compute embedding token
                self.doc.field_title => document.title,
                self.doc.field_link => document.link,
                self.doc.field_snippet => document.snippet,
            })
            .expect("Failed to add document");
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
