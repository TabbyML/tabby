use std::{fs, io::IsTerminal, path::Path};

use kdam::BarExt;
use tabby_common::{
    config::RepositoryConfig,
    index::{register_tokenizers, CodeSearchSchema},
    path, SourceFile,
};
use tantivy::{directory::MmapDirectory, doc, Index};
use tracing::{debug, warn};

use crate::{code::CodeIntelligence, utils::tqdm};

// Magic numbers
static MAX_LINE_LENGTH_THRESHOLD: usize = 300;
static AVG_LINE_LENGTH_THRESHOLD: f32 = 150f32;

pub fn index_repositories(_config: &[RepositoryConfig]) {
    let code = CodeSearchSchema::new();

    let index = open_or_create_index(&code, &path::index_dir());
    register_tokenizers(&index);

    // Initialize the search index writer with an initial arena size of 150 MB.
    let mut writer = index
        .writer(150_000_000)
        .expect("Failed to create index writer");
    writer
        .delete_all_documents()
        .expect("Failed to delete all documents");

    let total_file_size: usize = SourceFile::all()
        .filter(is_valid_file)
        .map(|x| x.read_file_size())
        .sum();

    let mut pb = std::io::stdout()
        .is_terminal()
        .then(|| tqdm(total_file_size));

    let intelligence = CodeIntelligence::default();
    for file in SourceFile::all().filter(is_valid_file) {
        let text = match file.read_content() {
            Ok(content) => content,
            Err(e) => {
                warn!("Failed to read content of '{}': {}", file.filepath, e);
                continue;
            }
        };

        for body in intelligence.chunks(&text) {
            pb.as_mut()
                .map(|b| b.update(body.len()))
                .transpose()
                .expect("Failed to update progress bar");

            writer
                .add_document(doc!(
                        code.field_git_url => file.git_url.clone(),
                        code.field_filepath => file.filepath.clone(),
                        code.field_language => file.language.clone(),
                        code.field_body => body,
                ))
                .expect("Failed to add document");
        }
    }

    writer.commit().expect("Failed to commit index");
    writer
        .wait_merging_threads()
        .expect("Failed to wait for merging threads");
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
