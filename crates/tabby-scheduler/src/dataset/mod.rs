mod deps;

use std::{
    fs::{self},
    io::{IsTerminal, Write},
};

use file_rotate::{compression::Compression, suffix::AppendCount, ContentLimit, FileRotate};

use kdam::BarExt;
use serde_jsonlines::WriteExt;
use tabby_common::{
    config::RepositoryConfig,
    path::{dataset_dir, dependency_file},
    DependencyFile, SourceFile,
};

use crate::{cache::CacheStore, utils::tqdm};

pub fn create_dataset(cache: &CacheStore, config: &[RepositoryConfig]) {
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
    for repository in config {
        deps::collect(repository.dir().as_path(), &mut deps);
    }
    let mut pb = Some(cache.source_files().count())
        .filter(|_| std::io::stdout().is_terminal())
        .map(tqdm);

    for source_file in cache.source_files() {
        writer
            .write_json_lines([source_file.clone()])
            .expect("Failed to write dataset jsonl file");
        pb.as_mut()
            .map(|b| b.update(1))
            .transpose()
            .expect("Failed to update progress bar");
    }

    serdeconv::to_json_file(&deps, dependency_file())
        .expect("Failed to write dependencies json file");

    writer.flush().expect("Failed to flush writer");
}
