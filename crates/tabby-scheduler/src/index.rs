use std::{fs, io::IsTerminal, path::Path};

use ignore::Walk;
use kdam::BarExt;
use tabby_common::{
    config::RepositoryConfig,
    index::{register_tokenizers, CodeSearchSchema},
    languages::get_language_by_ext,
    path, SourceFile,
};
use tantivy::{directory::MmapDirectory, doc, Index, IndexWriter, Term};
use tracing::{debug, warn};

use crate::{
    cache::{get_changed_files, get_current_commit_hash, CacheStore},
    code::CodeIntelligence,
    utils::tqdm,
};

// Magic numbers
static MAX_LINE_LENGTH_THRESHOLD: usize = 300;
static AVG_LINE_LENGTH_THRESHOLD: f32 = 150f32;

pub fn index_repositories(config: &[RepositoryConfig]) {
    let code = CodeSearchSchema::new();
    let mut cache = CacheStore::new(tabby_common::path::cache_dir());

    let index = open_or_create_index(&code, &path::index_dir());
    register_tokenizers(&index);

    // Initialize the search index writer with an initial arena size of 150 MB.
    let mut writer = index
        .writer(150_000_000)
        .expect("Failed to create index writer");

    let intelligence = CodeIntelligence::default();

    writer
        .delete_all_documents()
        .expect("Failed to delete all documents");

    for repository in config {
        let Some(commit) = cache.get_last_index_commit(repository) else {
            index_repository_from_scratch(repository, &writer, &code, &intelligence, &mut cache);
            continue;
        };
        let dir = repository.dir();
        let changed_files = get_changed_files(&dir, &commit).expect("Failed read file diff");
        for file in changed_files {
            let path = dir.join(&file);
            delete_indexed_source_file(&writer, &code, &repository.git_url, &file);
            if !path.exists() {
                continue;
            }
            let Some(source_file) = cache.get_source_file(repository, &path) else {
                continue;
            };
            if !is_valid_file(&source_file) {
                continue;
            }
            add_indexed_source_file(&writer, repository, &source_file, &code, &intelligence);
        }
        cache.set_last_index_commit(
            repository,
            Some(get_current_commit_hash(&dir).expect("Failed to read current commit hash")),
        );
    }

    for indexed_repository in cache.list_indexed_repositories() {
        if !indexed_repository.dir().exists() {
            cache.set_last_index_commit(&indexed_repository, None);
            delete_all_indexed_files(&writer, &code, &indexed_repository.canonical_git_url());
        }
    }

    writer.commit().expect("Failed to commit index");
    writer
        .wait_merging_threads()
        .expect("Failed to wait for merging threads");
}

fn index_repository_from_scratch(
    repository: &RepositoryConfig,
    writer: &IndexWriter,
    code: &CodeSearchSchema,
    intelligence: &CodeIntelligence,
    cache: &mut CacheStore,
) {
    let mut pb = std::io::stdout().is_terminal().then(|| {
        let total_file_size: usize = Walk::new(repository.dir())
            .filter_map(|f| f.ok())
            .map(|f| f.path().to_owned())
            .filter(|f| {
                f.extension()
                    .is_some_and(|ext| get_language_by_ext(ext).is_some())
            })
            .map(|f| read_file_size(&f))
            .sum();
        tqdm(total_file_size)
    });

    for file in Walk::new(repository.dir()) {
        let file = file.expect("Failed to read file listing");
        let Some(source_file) = cache.get_source_file(repository, file.path()) else {
            continue;
        };
        if !is_valid_file(&source_file) {
            continue;
        }
        add_indexed_source_file(writer, repository, &source_file, code, intelligence);
        pb.as_mut().map(|pb| {
            pb.update(source_file.read_file_size())
                .expect("Failed to update progress bar")
        });
    }
    cache.set_last_index_commit(
        repository,
        Some(
            get_current_commit_hash(&repository.dir()).expect("Failed to read current commit hash"),
        ),
    );
}

fn read_file_size(path: &Path) -> usize {
    std::fs::metadata(path)
        .map(|meta| meta.len())
        .unwrap_or_default() as usize
}

fn is_valid_file(source_file: &SourceFile) -> bool {
    source_file.max_line_length <= MAX_LINE_LENGTH_THRESHOLD
        && source_file.avg_line_length <= AVG_LINE_LENGTH_THRESHOLD
}

pub fn delete_indexed_source_file(
    writer: &IndexWriter,
    code: &CodeSearchSchema,
    git_url: &str,
    filepath: &str,
) {
    let file_id = SourceFile::create_file_id(git_url, filepath);
    let term = Term::from_field_text(code.field_file_id.clone(), &file_id);
    writer.delete_term(term);
}

pub fn delete_all_indexed_files(writer: &IndexWriter, code: &CodeSearchSchema, git_url: &str) {
    let term = Term::from_field_text(code.field_git_url, git_url);
    writer.delete_term(term);
}

pub fn add_indexed_source_file(
    writer: &IndexWriter,
    repository: &RepositoryConfig,
    file: &SourceFile,
    code: &CodeSearchSchema,
    intelligence: &CodeIntelligence,
) -> usize {
    let text = match file.read_content() {
        Ok(content) => content,
        Err(e) => {
            warn!("Failed to read content of '{}': {}", file.filepath, e);
            return 0;
        }
    };
    for body in intelligence.chunks(&text) {
        writer
            .add_document(doc!(
                    code.field_git_url => repository.canonical_git_url(),
                    code.field_filepath => file.filepath.clone(),
                    code.field_file_id => SourceFile::create_file_id(&repository.git_url, &file.filepath),
                    code.field_language => file.language.clone(),
                    code.field_body => body,
            ))
            .expect("Failed to add document");
    }
    text.len()
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
