use std::{fs, io::IsTerminal, path::Path};

use ignore::Walk;
use kdam::BarExt;
use kv::Batch;
use tabby_common::{
    config::RepositoryConfig,
    index::{register_tokenizers, CodeSearchSchema},
    path, SourceFile,
};
use tantivy::{directory::MmapDirectory, doc, Index, Term};
use tracing::{debug, warn};

use crate::{cache::CacheStore, code::CodeIntelligence, utils::tqdm};

// Magic numbers
static MAX_LINE_LENGTH_THRESHOLD: usize = 300;
static AVG_LINE_LENGTH_THRESHOLD: f32 = 150f32;

pub fn index_repositories(cache: &mut CacheStore, config: &[RepositoryConfig]) {
    let code = CodeSearchSchema::new();

    let index = open_or_create_index(&code, &path::index_dir());
    register_tokenizers(&index);

    // Initialize the search index writer with an initial arena size of 150 MB.
    let mut writer = index
        .writer(150_000_000)
        .expect("Failed to create index writer");

    let total_file_size: usize = SourceFile::all()
        .filter(is_valid_file)
        .map(|x| x.read_file_size())
        .sum();

    let mut pb = std::io::stdout()
        .is_terminal()
        .then(|| tqdm(total_file_size));

    let intelligence = CodeIntelligence::default();
    let mut indexed_files_batch = Batch::new();
    for repository in config {
        for file in Walk::new(repository.dir()) {
            let file = match file {
                Ok(file) => file,
                Err(e) => {
                    warn!("Failed to walk file tree for indexing: {e}");
                    continue;
                }
            };
            let Some(source_file) = cache.get_source_file(repository, file.path()) else {
                continue;
            };
            if !is_valid_file(&source_file) {
                continue;
            }
            let (file_id, indexed) = cache.check_indexed(file.path());

            if indexed {
                continue;
            }
            let text = match source_file.read_content() {
                Ok(content) => content,
                Err(e) => {
                    warn!(
                        "Failed to read content of '{}': {}",
                        source_file.filepath, e
                    );
                    continue;
                }
            };

            for body in intelligence.chunks(&text) {
                pb.as_mut()
                    .map(|b| b.update(body.len()))
                    .transpose()
                    .expect("Failed to update progress bar");

                writer
                    .add_document(doc! {
                                code.field_git_url => source_file.git_url.clone(),
                                code.field_source_file_key => file_id.to_string(),
                                code.field_filepath => source_file.filepath.clone(),
                                code.field_language => source_file.language.clone(),
                                code.field_body => body,
                    })
                    .expect("Failed to add document");
            }
            indexed_files_batch
                .set(&file_id, &String::new())
                .expect("Failed to mark file as indexed");
        }
    }

    // Commit updating indexed documents
    writer.commit().expect("Failed to commit index");
    writer
        .wait_merging_threads()
        .expect("Failed to wait for merging threads");

    // Mark all indexed documents as indexed
    cache.set_indexed(indexed_files_batch);

    // Create a new writer to commit deletion of removed indexed files
    let mut writer = index
        .writer(150_000_000)
        .expect("Failed to create index writer");

    cache.garbage_collection_for_indexed_files(|key| {
        writer.delete_term(Term::from_field_text(code.field_source_file_key, key));
    });

    // Commit garbage collection
    writer
        .commit()
        .expect("Failed to commit garbage collection");

    writer
        .wait_merging_threads()
        .expect("Failed to wait for merging threads on garbage collection");
}

fn is_valid_file(file: &SourceFile) -> bool {
    file.max_line_length <= MAX_LINE_LENGTH_THRESHOLD
        && file.avg_line_length <= AVG_LINE_LENGTH_THRESHOLD
}

fn open_or_create_index(code: &CodeSearchSchema, path: &Path) -> Index {
    match open_or_create_index_impl(code, path) {
        Ok(index) => index,
        Err(err) => {
            warn!(
                "Failed to open index repositories: {}, removing index directory '{}'...",
                err,
                path.display()
            );
            fs::remove_dir_all(path).expect("Failed to remove index directory");

            debug!("Reopening index repositories...");
            open_or_create_index_impl(code, path).expect("Failed to open index")
        }
    }
}

fn open_or_create_index_impl(code: &CodeSearchSchema, path: &Path) -> tantivy::Result<Index> {
    fs::create_dir_all(path).expect("Failed to create index directory");
    let directory = MmapDirectory::open(path).expect("Failed to open index directory");
    Index::open_or_create(directory, code.schema.clone())
}
