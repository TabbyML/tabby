mod deps;

use std::{
    fs::{self, read_to_string},
    io::{IsTerminal, Write},
};

use anyhow::{anyhow, Result};
use file_rotate::{compression::Compression, suffix::AppendCount, ContentLimit, FileRotate};
use ignore::{DirEntry, Walk};
use kdam::BarExt;
use serde_jsonlines::WriteExt;
use tabby_common::{
    config::RepositoryConfig,
    languages::get_language_by_ext,
    path::{dataset_dir, dependency_file},
    DependencyFile, SourceFile,
};
use tracing::error;

use crate::{code::CodeIntelligence, utils::tqdm};

trait RepositoryExt {
    fn create_dataset(&self, writer: &mut impl Write) -> Result<()>;
}

impl RepositoryExt for RepositoryConfig {
    fn create_dataset(&self, writer: &mut impl Write) -> Result<()> {
        let basedir = self.dir();
        let walk_dir_iter = || {
            Walk::new(basedir.as_path())
                .filter_map(Result::ok)
                .filter(is_source_code)
        };

        let mut pb = std::io::stdout()
            .is_terminal()
            .then(|| tqdm(walk_dir_iter().count()));
        let walk_dir = walk_dir_iter();

        let mut code = CodeIntelligence::default();
        for entry in walk_dir {
            pb.as_mut().map(|b| b.update(1)).transpose()?;

            let relative_path = entry
                .path()
                .strip_prefix(basedir.as_path())
                .expect("Paths always begin with the prefix");
            let language = get_language_by_ext(
                relative_path
                    .extension()
                    .ok_or_else(|| anyhow!("Unknown file extension for {relative_path:?}"))?,
            )
            .ok_or_else(|| anyhow!("Unknown language for {relative_path:?}"))?
            .to_owned()
            .language();
            match read_to_string(entry.path()) {
                Ok(file_content) => {
                    let source_file = SourceFile {
                        git_url: self.canonical_git_url(),
                        basedir: basedir.display().to_string(),
                        filepath: relative_path.display().to_string(),
                        max_line_length: metrics::max_line_length(&file_content),
                        avg_line_length: metrics::avg_line_length(&file_content),
                        alphanum_fraction: metrics::alphanum_fraction(&file_content),
                        tags: code.find_tags(language, &file_content),
                        language: language.into(),
                    };
                    writer.write_json_lines([source_file.clone()])?;
                }
                Err(e) => {
                    error!(
                        "Cannot read '{}/{}': '{e}'",
                        basedir.display(),
                        relative_path.display()
                    );
                }
            }
        }

        Ok(())
    }
}

fn is_source_code(entry: &DirEntry) -> bool {
    if entry.file_type().is_some_and(|x| x.is_file()) {
        entry
            .path()
            .extension()
            .and_then(get_language_by_ext)
            .is_some()
    } else {
        false
    }
}

pub fn create_dataset(config: &[RepositoryConfig]) -> Result<()> {
    fs::remove_dir_all(dataset_dir()).ok();
    fs::create_dir_all(dataset_dir())?;

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
        repository.create_dataset(&mut writer)?;
    }

    serdeconv::to_json_file(&deps, dependency_file())?;

    writer.flush()?;
    Ok(())
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
