mod deps;
mod tags;

use std::{
    collections::HashMap,
    ffi::OsStr,
    fs::{self, read_to_string},
    io::{IsTerminal, Write},
};

use anyhow::Result;
use file_rotate::{compression::Compression, suffix::AppendCount, ContentLimit, FileRotate};
use ignore::{DirEntry, Walk};
use kdam::BarExt;
use lazy_static::lazy_static;
use serde_jsonlines::WriteExt;
use tabby_common::{
    config::{Config, RepositoryConfig},
    path::{dataset_dir, dependency_file},
    DependencyFile, SourceFile,
};
use tracing::error;
use tree_sitter_tags::TagsContext;

use crate::utils::tqdm;

trait RepositoryExt {
    fn create_dataset(&self, writer: &mut impl Write) -> Result<()>;
}

impl RepositoryExt for RepositoryConfig {
    fn create_dataset(&self, writer: &mut impl Write) -> Result<()> {
        let dir = self.dir();

        let walk_dir_iter = || {
            Walk::new(dir.as_path())
                .filter_map(Result::ok)
                .filter(is_source_code)
        };

        let mut pb = std::io::stdout()
            .is_terminal()
            .then(|| tqdm(walk_dir_iter().count()));
        let walk_dir = walk_dir_iter();

        let mut context = TagsContext::new();
        for entry in walk_dir {
            pb.as_mut().map(|b| b.update(1)).transpose()?;

            let relative_path = entry.path().strip_prefix(dir.as_path()).unwrap();
            let language = get_language(relative_path.extension().unwrap())
                .unwrap()
                .to_owned();
            if let Ok(file_content) = read_to_string(entry.path()) {
                let source_file = SourceFile {
                    git_url: self.git_url.clone(),
                    filepath: relative_path.display().to_string(),
                    max_line_length: metrics::max_line_length(&file_content),
                    avg_line_length: metrics::avg_line_length(&file_content),
                    alphanum_fraction: metrics::alphanum_fraction(&file_content),
                    tags: tags::collect(&mut context, &language, &file_content),
                    language,
                    content: file_content,
                };
                writer.write_json_lines([source_file])?;
            } else {
                error!("Cannot read {:?}", relative_path);
            }
        }

        Ok(())
    }
}

fn get_language(ext: &OsStr) -> Option<&str> {
    let ext = ext.to_str().unwrap_or("");
    EXTENSION_LANGUAGE.get(ext).copied()
}

fn is_source_code(entry: &DirEntry) -> bool {
    if entry.file_type().is_some_and(|x| x.is_file()) {
        entry.path().extension().and_then(get_language).is_some()
    } else {
        false
    }
}

pub fn create_dataset(config: &Config) -> Result<()> {
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
    for repository in config.repositories.as_slice() {
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

lazy_static! {
    static ref LANGUAGE_EXTENSION: HashMap<&'static str, Vec<&'static str>> = {
        HashMap::from([
            ("c", vec!["c", "h"]),
            ("csharp", vec!["cs"]),
            (
                "cpp",
                vec!["cpp", "hpp", "c++", "h++", "cc", "hh", "C", "H", "tcc"],
            ),
            ("css", vec!["css"]),
            ("dockerfile", vec!["Dockerfile"]),
            ("go", vec!["go"]),
            ("haskell", vec!["hs"]),
            ("html", vec!["html"]),
            ("java", vec!["java"]),
            ("kotlin", vec!["kt", "kts"]),
            ("julia", vec!["jl"]),
            ("lua", vec!["lua"]),
            ("makefile", vec!["Makefile"]),
            ("markdown", vec!["md", "markdown"]),
            ("php", vec!["php", "php3", "php4", "php5", "phps", "phpt"]),
            ("perl", vec!["pl", "pm", "pod", "perl"]),
            ("powershell", vec!["ps1", "psd1", "psm1"]),
            ("python", vec!["py"]),
            ("ruby", vec!["rb"]),
            ("rust", vec!["rs"]),
            ("sql", vec!["sql"]),
            ("scala", vec!["scala"]),
            ("shellscript", vec!["sh", "bash", "command", "zsh"]),
            (
                "javascript-typescript",
                vec!["ts", "mts", "js", "mjs", "jsx", "tsx"],
            ),
            ("tex", vec!["tex"]),
            ("vb", vec!["vb"]),
        ])
    };
    static ref EXTENSION_LANGUAGE: HashMap<&'static str, &'static str> = {
        let mut map = HashMap::new();
        for (lang, exts) in &*LANGUAGE_EXTENSION {
            for ext in exts {
                map.insert(*ext, *lang);
            }
        }

        map
    };
}
