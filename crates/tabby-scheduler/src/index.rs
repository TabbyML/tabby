use std::fs;

use anyhow::Result;
use tabby_common::{config::Config, path::index_dir, Document};
use tantivy::{
    directory::MmapDirectory,
    doc,
    schema::{Schema, STORED, STRING, TEXT},
    Index,
};

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
        writer.add_document(doc!(
                git_url => doc.git_url,
                filepath => doc.filepath,
                content => doc.content,
                language => doc.language,
        ))?;
    }

    writer.commit()?;

    Ok(())
}
