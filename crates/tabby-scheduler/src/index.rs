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

use crate::utils::reduce_language_if_needed;

pub fn index_repositories(_config: &Config) -> Result<()> {
    let mut builder = Schema::builder();

    let field_git_url = builder.add_text_field("git_url", STRING | STORED);
    let field_filepath = builder.add_text_field("filepath", STRING | STORED);
    let field_language = builder.add_text_field("language", STRING | STORED);
    let field_name = builder.add_text_field("name", STRING | STORED);
    let field_kind = builder.add_text_field("kind", STRING | STORED);
    let field_body = builder.add_text_field("body", TEXT | STORED);

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

            let language = reduce_language_if_needed(doc.language.as_str());
            writer.add_document(doc!(
                    field_git_url => doc.git_url.clone(),
                    field_filepath => doc.filepath.clone(),
                    field_language => language,
                    field_name => name,
                    field_body => body,
                    field_kind => tag.syntax_type_name,
            ))?;
        }
    }

    writer.commit()?;

    Ok(())
}

lazy_static! {
    static ref LANGUAGE_NAME_BLACKLIST: HashMap<&'static str, Vec<&'static str>> =
        HashMap::from([("python", vec!["__init__"])]);
}
