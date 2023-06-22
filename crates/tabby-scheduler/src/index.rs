use std::fs;

use anyhow::Result;
use tabby_common::{config::Config, path::index_dir, Document};
use tantivy::{
    directory::MmapDirectory,
    doc,
    schema::{Schema, STORED, STRING, TEXT},
    Index,
};
use tracing::info;

pub fn index_repositories(_config: &Config) -> Result<()> {
    let mut builder = Schema::builder();
    let git_url = builder.add_text_field("git_url", STRING | STORED);
    let filepath = builder.add_text_field("filepath", STRING | STORED);
    let content = builder.add_text_field("content", TEXT | STORED);
    let language = builder.add_text_field("language", TEXT | STORED);
    let schema = builder.build();

    fs::create_dir_all(index_dir())?;
    let directory = MmapDirectory::open(index_dir())?;
    let index = Index::open_or_create(directory, schema)?;
    let mut writer = index.writer(10_000_000)?;
    writer.delete_all_documents()?;

    for doc in Document::all()? {
        if is_valid_doc(&doc) {
            writer.add_document(doc!(
                    git_url => doc.git_url,
                    filepath => doc.filepath,
                    content => doc.content,
                    language => doc.language,
            ))?;
        } else {
            info!("Skip {} - {}", doc.git_url, doc.filepath);
        }
    }

    info!("Finalize index...");
    writer.commit()?;

    Ok(())
}

fn is_valid_doc(x: &Document) -> bool {
    if x.max_line_length > 1000 {
        false
    } else if x.avg_line_length > 100.0 {
        false
    } else {
        x.alphanum_fraction >= 0.25
    }
}
