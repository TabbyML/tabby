mod deps;

use std::{
    fs::{self},
    io::{IsTerminal, Write},
};

use file_rotate::{compression::Compression, suffix::AppendCount, ContentLimit, FileRotate};
use ignore::Walk;
use kdam::BarExt;
use serde_jsonlines::WriteExt;
use tabby_common::{
    config::RepositoryConfig,
    path::{self, dataset_dir, dependency_file},
    DependencyFile, SourceFile,
};

use crate::{repository_store::RepositoryStore, utils::tqdm};

fn export_json_dataset(
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
    let repository_store = RepositoryStore::new(tabby_common::path::repository_store());
    repository_store.update_dataset(config);

    for repository in config {
        deps::collect(repository.dir().as_path(), &mut deps);
    }
    export_json_dataset(
        repository_store.source_files(),
        &mut writer,
        Some(Walk::new(path::repositories_dir()).count()),
    );

    serdeconv::to_json_file(&deps, dependency_file())
        .expect("Failed to write dependencies json file");

    writer.flush().expect("Failed to flush writer");
}
