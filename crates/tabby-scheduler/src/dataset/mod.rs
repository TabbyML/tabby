mod deps;

use std::{
    fs::{self},
    io::{IsTerminal, Write},
};

use anyhow::Result;
use file_rotate::{compression::Compression, suffix::AppendCount, ContentLimit, FileRotate};
use ignore::Walk;
use kdam::BarExt;
use serde_jsonlines::WriteExt;
use tabby_common::{
    config::RepositoryConfig,
    path::{dataset_dir, dependency_file},
    DependencyFile, SourceFile,
};

use crate::{cache::CacheStore, utils::tqdm};

trait RepositoryExt {
    fn create_dataset(&self, cache: &mut CacheStore, writer: &mut impl Write) -> Result<()>;
}

impl RepositoryExt for RepositoryConfig {
    fn create_dataset(&self, cache: &mut CacheStore, writer: &mut impl Write) -> Result<()> {
        let dir = self.dir();

        let walk_dir_iter = || Walk::new(dir.as_path()).filter_map(Result::ok);

        let mut pb = std::io::stdout()
            .is_terminal()
            .then(|| tqdm(walk_dir_iter().count()));

        for entry in walk_dir_iter() {
            let Some(source_file) = cache.get_or_create_source_file(self, entry.path()) else {
                continue;
            };
            writer
                .write_json_lines([source_file.clone()])
                .expect("Failed to write dataset jsonl file");
            pb.as_mut()
                .map(|b| b.update(1))
                .transpose()
                .expect("Failed to update progress bar");
        }

        Ok(())
    }
}

pub fn create_dataset(cache: &mut CacheStore, config: &[RepositoryConfig]) {
    fs::remove_dir_all(dataset_dir()).ok();
    fs::create_dir_all(dataset_dir()).expect("Failed to create dataset directory");

    // Collect dependencies
    {
        let mut deps = DependencyFile::default();
        for repository in config {
            deps::collect(repository.dir().as_path(), &mut deps);
        }
        serdeconv::to_json_file(&deps, dependency_file())
            .expect("Failed to write dependencies json file");
    }

    // Create dataset
    let mut writer = FileRotate::new(
        SourceFile::files_jsonl(),
        AppendCount::new(usize::max_value()),
        ContentLimit::Lines(1000),
        Compression::None,
        #[cfg(unix)]
        None,
    );
    for repository in config {
        repository
            .create_dataset(cache, &mut writer)
            .expect("Failed to create dataset");
    }

    writer.flush().expect("Failed to flush writer");
}
