use std::fs::{self, read_to_string};

use anyhow::Result;
use tabby_common::{
    config::{Config, Repository},
    path::index_dir,
};
use tantivy::{
    directory::MmapDirectory,
    doc,
    schema::{Schema, STORED, STRING, TEXT},
    Index, IndexWriter,
};
use tracing::{info, warn};
use walkdir::{DirEntry, WalkDir};

trait RepositoryExt {
    fn index(&self, schema: &Schema, writer: &mut IndexWriter) -> Result<()>;
}

impl RepositoryExt for Repository {
    fn index(&self, schema: &Schema, writer: &mut IndexWriter) -> Result<()> {
        let git_url = schema.get_field("git_url").unwrap();
        let filepath = schema.get_field("filepath").unwrap();
        let content = schema.get_field("content").unwrap();
        let dir = self.dir();

        info!("Start indexing repository {}", self.git_url);
        let walk_dir = WalkDir::new(dir.as_path())
            .into_iter()
            .filter_entry(is_not_hidden)
            .filter_map(Result::ok)
            .filter(|e| !e.file_type().is_dir());

        for entry in walk_dir {
            let relative_path = entry.path().strip_prefix(dir.as_path()).unwrap();
            if let Ok(file_content) = read_to_string(entry.path()) {
                info!("Indexing {:?}", relative_path);
                writer.add_document(doc!(
                    git_url => self.git_url.clone(),
                    filepath => relative_path.display().to_string(),
                    content => file_content,
                ))?;
            } else {
                warn!("Skip {:?}", relative_path);
            }
        }

        Ok(())
    }
}

fn is_not_hidden(entry: &DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .map(|s| entry.depth() == 0 || !s.starts_with('.'))
        .unwrap_or(false)
}

fn create_schema() -> Schema {
    let mut builder = Schema::builder();
    builder.add_text_field("git_url", STRING | STORED);
    builder.add_text_field("filepath", STRING | STORED);
    builder.add_text_field("content", TEXT | STORED);
    builder.build()
}

pub fn index_repositories(config: &Config) -> Result<()> {
    let schema = create_schema();

    fs::create_dir_all(index_dir())?;
    let directory = MmapDirectory::open(index_dir())?;
    let index = Index::open_or_create(directory, schema.clone())?;
    let mut writer = index.writer(10_000_000)?;

    writer.delete_all_documents()?;
    for repository in config.repositories.as_slice() {
        repository.index(&schema, &mut writer)?;
    }

    writer.commit()?;

    Ok(())
}
