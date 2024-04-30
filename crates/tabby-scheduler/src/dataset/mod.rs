mod deps;

use std::{
    fs::{self, read_to_string},
    io::{IsTerminal, Write},
    path::Path,
};

use anyhow::{anyhow, Result};
use file_rotate::{compression::Compression, suffix::AppendCount, ContentLimit, FileRotate};
use ignore::Walk;
use kdam::BarExt;
use serde_jsonlines::WriteExt;
use tabby_common::{
    config::RepositoryConfig,
    languages::get_language_by_ext,
    path::{self, dataset_dir, dependency_file},
    DependencyFile, SourceFile,
};
use tracing::debug;

use crate::{code::CodeIntelligence, repository_store::RepositoryStore, utils::tqdm};

pub fn build_repository_dataset(
    repository: &RepositoryConfig,
) -> impl Iterator<Item = SourceFile> + '_ {
    let basedir = repository.dir();
    let walk_dir = Walk::new(basedir.as_path()).filter_map(Result::ok);

    let mut code = CodeIntelligence::default();
    walk_dir.filter_map(move |entry| create_source_file(repository, entry.path(), &mut code))
}

pub fn dump_json_dataset(
    dataset: impl Iterator<Item = SourceFile>,
    writer: &mut impl Write,
    item_count: Option<usize>,
) {
    let mut pb = item_count
        .filter(|_| std::io::stdout().is_terminal())
        .map(tqdm);

    for source_file in dataset {
        writer
            .write_json_lines([source_file.clone()])
            .expect("Failed to write dataset jsonl file");
        pb.as_mut()
            .map(|b| b.update(1))
            .transpose()
            .expect("Failed to update progress bar");
    }
}

pub fn create_source_file(
    config: &RepositoryConfig,
    path: &Path,
    code: &mut CodeIntelligence,
) -> Option<SourceFile> {
    if path.is_dir() || !path.exists() {
        return None;
    }
    let relative_path = path
        .strip_prefix(&config.dir())
        .expect("Paths always begin with the prefix");

    let Some(ext) = relative_path.extension() else {
        return None;
    };

    let Some(language_info) = get_language_by_ext(ext) else {
        debug!("Unknown language for {relative_path:?}");
        return None;
    };

    let language = language_info.language();
    let contents = read_to_string(path)
        .map_err(|e| anyhow!("Failed to read {path:?}: {e}"))
        .unwrap();
    let source_file = SourceFile {
        git_url: config.canonical_git_url(),
        basedir: config.dir().display().to_string(),
        filepath: relative_path.display().to_string(),
        max_line_length: metrics::max_line_length(&contents),
        avg_line_length: metrics::avg_line_length(&contents),
        alphanum_fraction: metrics::alphanum_fraction(&contents),
        tags: code.find_tags(language, &contents),
        language: language.into(),
    };
    Some(source_file)
}

pub fn create_dataset(config: &[RepositoryConfig]) {
    fs::remove_dir_all(dataset_dir()).ok();
    fs::create_dir_all(dataset_dir()).expect("Failed to create dataset directory");

    let mut writer = FileRotate::new(
        SourceFile::files_jsonl(),
        AppendCount::new(usize::max_value()),
        ContentLimit::Lines(1000),
        Compression::None,
        #[cfg(unix)]
        None,
    );

    let mut deps = DependencyFile::default();
    let repository_store = RepositoryStore::new();
    repository_store.update_dataset(config);

    for repository in config {
        deps::collect(repository.dir().as_path(), &mut deps);
    }
    dump_json_dataset(
        repository_store.cached_source_files(),
        &mut writer,
        Some(Walk::new(path::repositories_dir()).count()),
    );

    serdeconv::to_json_file(&deps, dependency_file())
        .expect("Failed to write dependencies json file");

    writer.flush().expect("Failed to flush writer");
}

mod metrics {
    use std::cmp::max;

    pub fn max_line_length(content: &str) -> usize {
        content.lines().map(|x| x.len()).reduce(max).unwrap_or(0)
    }

    pub fn avg_line_length(content: &str) -> f32 {
        let mut total = 0;
        let mut len = 0;
        for x in content.lines() {
            len += 1;
            total += x.len();
        }

        if len > 0 {
            total as f32 / len as f32
        } else {
            0.0
        }
    }

    pub fn alphanum_fraction(content: &str) -> f32 {
        let num_alphanumn: f32 = content
            .chars()
            .map(|x| f32::from(u8::from(x.is_alphanumeric())))
            .sum();
        if !content.is_empty() {
            num_alphanumn / content.len() as f32
        } else {
            0.0
        }
    }
}
