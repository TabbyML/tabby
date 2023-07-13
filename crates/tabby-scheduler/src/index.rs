use std::{collections::HashMap, fs};

use anyhow::Result;
use lazy_static::lazy_static;
use tabby_common::{config::Config, path::index_dir, Document};
use tantivy::{
    directory::MmapDirectory,
    doc,
    schema::{Schema, STORED, STRING, TEXT},
    Index,
};

pub fn index_repositories(_config: &Config) -> Result<()> {
    let mut builder = Schema::builder();
    let git_url = builder.add_text_field("name", STRING | STORED);
    let filepath = builder.add_text_field("body", STRING | STORED);
    let content = builder.add_text_field("content", TEXT | STORED);
    let language = builder.add_text_field("language", TEXT | STORED);
    let schema = builder.build();

    fs::create_dir_all(index_dir())?;
    let directory = MmapDirectory::open(index_dir())?;
    let index = Index::open_or_create(directory, schema)?;
    let mut writer = index.writer(10_000_000)?;
    writer.delete_all_documents()?;

    for doc in Document::all()? {
        for tag in doc.tags {
            let name = doc.content.get(tag.name_range).unwrap();
            if name.len() < 5 {
                continue;
            }

            let body = doc.content.get(tag.range).unwrap();
            let count_body_lines = body.lines().count();
            if !(3..=10).contains(&count_body_lines) {
                continue;
            }

            if let Some(blacklist) = LANGUAGE_NAME_BLACKLIST.get(doc.language.as_str()) {
                if blacklist.contains(&name) {
                    continue;
                }
            }
        }
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

lazy_static! {
    static ref LANGUAGE_NAME_BLACKLIST: HashMap<&'static str, Vec<&'static str>> =
        HashMap::from([("python", vec!["__init__"])]);
}
