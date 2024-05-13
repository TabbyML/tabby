use ignore::Walk;
use kv::Batch;
use tabby_common::{config::RepositoryConfig, index::CodeSearchSchema, path};
use tantivy::{doc, Index, Term};
use tracing::warn;

use super::{
    cache::CacheStore,
    intelligence::{CodeIntelligence, SourceFile},
};
use crate::tantivy_utils::open_or_create_index;

// Magic numbers
static MAX_LINE_LENGTH_THRESHOLD: usize = 300;
static AVG_LINE_LENGTH_THRESHOLD: f32 = 150f32;

pub fn index_repository(cache: &mut CacheStore, repository: &RepositoryConfig) {
    let code = CodeSearchSchema::default();
    let index = open_or_create_index(&code.schema, &path::index_dir());
    add_changed_documents(cache, &code, repository, &index);
}

pub fn garbage_collection(cache: &mut CacheStore) {
    let code = CodeSearchSchema::default();
    let index = open_or_create_index(&code.schema, &path::index_dir());
    remove_staled_documents(cache, &code, &index);
}

fn add_changed_documents(
    cache: &mut CacheStore,
    code: &CodeSearchSchema,
    repository: &RepositoryConfig,
    index: &Index,
) {
    // Initialize the search index writer with an initial arena size of 150 MB.
    let mut writer = index
        .writer(150_000_000)
        .expect("Failed to create index writer");

    let intelligence = CodeIntelligence::default();
    let mut indexed_files_batch = Batch::new();
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

    // Commit updating indexed documents
    writer.commit().expect("Failed to commit index");
    writer
        .wait_merging_threads()
        .expect("Failed to wait for merging threads");

    // Mark all indexed documents as indexed
    cache.apply_indexed(indexed_files_batch);
}

pub fn remove_staled_documents(cache: &mut CacheStore, code: &CodeSearchSchema, index: &Index) {
    // Create a new writer to commit deletion of removed indexed files
    let mut writer = index
        .writer(150_000_000)
        .expect("Failed to create index writer");

    let gc_commit = cache.prepare_garbage_collection_for_indexed_files(|key| {
        writer.delete_term(Term::from_field_text(code.field_source_file_key, key));
    });

    // Commit garbage collection
    writer
        .commit()
        .expect("Failed to commit garbage collection");

    writer
        .wait_merging_threads()
        .expect("Failed to wait for merging threads on garbage collection");

    gc_commit();
}

fn is_valid_file(file: &SourceFile) -> bool {
    file.max_line_length <= MAX_LINE_LENGTH_THRESHOLD
        && file.avg_line_length <= AVG_LINE_LENGTH_THRESHOLD
}
