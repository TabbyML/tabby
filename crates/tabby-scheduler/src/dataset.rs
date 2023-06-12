use std::{
    collections::HashMap,
    ffi::OsStr,
    fs::{self, read_to_string, File},
};

use anyhow::Result;
use lazy_static::lazy_static;
use serde::Serialize;
use serde_jsonlines::JsonLinesWriter;
use tabby_common::{
    config::{Config, Repository},
    path::dataset_dir,
};
use tracing::{error, info};
use walkdir::{DirEntry, WalkDir};

lazy_static! {
    static ref LANGUAGE_EXTENSION: HashMap<&'static str, Vec<&'static str>> = {
        HashMap::from([
            ("c", vec!["c", "h"]),
            ("csharp", vec!["cs"]),
            (
                "cpp",
                vec!["cpp", "hpp", "c++", "h++", "cc", "hh", "C", "H"],
            ),
            ("css", vec!["css"]),
            ("dockerfile", vec!["Dockerfile"]),
            ("go", vec!["go"]),
            ("haskell", vec!["hs"]),
            ("html", vec!["html"]),
            ("java", vec!["java"]),
            ("javascript", vec!["js"]),
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
            ("typescript", vec!["ts", "tsx"]),
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

trait RepositoryExt {
    fn create_dataset(&self, writer: &mut JsonLinesWriter<File>) -> Result<()>;
}

impl RepositoryExt for Repository {
    fn create_dataset(&self, writer: &mut JsonLinesWriter<File>) -> Result<()> {
        let dir = self.dir();

        info!("Start indexing repository {}", self.git_url);
        let walk_dir = WalkDir::new(dir.as_path())
            .into_iter()
            .filter_entry(is_not_hidden)
            .filter_map(Result::ok)
            .filter(is_source_code);

        for entry in walk_dir {
            let relative_path = entry.path().strip_prefix(dir.as_path()).unwrap();
            if let Ok(file_content) = read_to_string(entry.path()) {
                info!("Building {:?}", relative_path);
                writer.write(&Document {
                    git_url: self.git_url.clone(),
                    filepath: relative_path.display().to_string(),
                    content: file_content,
                    language: get_language(relative_path.extension().unwrap())
                        .unwrap()
                        .to_owned(),
                })?;
            } else {
                error!("Cannot read {:?}", relative_path);
            }
        }

        Ok(())
    }
}

#[derive(Serialize)]
struct Document {
    git_url: String,
    filepath: String,
    content: String,
    language: String,
}

fn get_language(ext: &OsStr) -> Option<&str> {
    let ext = ext.to_str().unwrap_or("");
    EXTENSION_LANGUAGE.get(ext).copied()
}

fn is_source_code(entry: &DirEntry) -> bool {
    if entry.file_type().is_file() {
        entry.path().extension().and_then(get_language).is_some()
    } else {
        false
    }
}

fn is_not_hidden(entry: &DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .map(|s| entry.depth() == 0 || !s.starts_with('.'))
        .unwrap_or(false)
}

pub fn create_dataset(config: &Config) -> Result<()> {
    fs::remove_dir_all(dataset_dir()).ok();
    fs::create_dir_all(dataset_dir())?;
    let file = File::create(dataset_dir().join("data.jsonl"))?;
    let mut writer = JsonLinesWriter::new(file);

    for repository in config.repositories.as_slice() {
        repository.create_dataset(&mut writer)?;
    }

    writer.flush()?;
    Ok(())
}
