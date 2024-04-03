mod deps;
mod tags;

use std::{
    collections::HashMap,
    ffi::OsStr,
    fs::read_to_string,
    io::{IsTerminal, Write},
};

use anyhow::{anyhow, Result};
use ignore::{DirEntry, Walk};
use kdam::BarExt;
use kdam::{tqdm, Bar};
use lazy_static::lazy_static;
use tabby_common::{config::RepositoryConfig, DependencyFile, SourceFile};
use tracing::error;
use tree_sitter_tags::TagsContext;

use crate::RepositoryCache;

trait RepositoryExt {
    fn create_dataset(&self, writer: &mut impl Write) -> Result<()>;
}

fn index_repository(cache: &RepositoryCache, repository: &RepositoryConfig) -> Result<()> {
    let dir = repository.dir();

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

        let relative_path = entry
            .path()
            .strip_prefix(dir.as_path())
            .expect("Paths always begin with the prefix");
        let language = get_language(
            relative_path
                .extension()
                .ok_or_else(|| anyhow!("Unknown file extension for {relative_path:?}"))?,
        )
        .ok_or_else(|| anyhow!("Unknown language for {relative_path:?}"))?
        .to_owned();
        match read_to_string(entry.path()) {
            Ok(file_content) => {
                let file = SourceFile {
                    git_url: repository.git_url.clone(),
                    repository_name: repository.name(),
                    filepath: relative_path.display().to_string(),
                    max_line_length: metrics::max_line_length(&file_content),
                    avg_line_length: metrics::avg_line_length(&file_content),
                    alphanum_fraction: metrics::alphanum_fraction(&file_content),
                    tags: tags::collect(&mut context, &language, &file_content),
                    language,
                    content: file_content,
                };
                cache.add_repository_meta(file)?;
            }
            Err(e) => {
                error!("Cannot read {relative_path:?}: {e:?}");
            }
        }
    }

    Ok(())
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

pub fn reload_index(cache: &RepositoryCache, config: &[RepositoryConfig]) -> Result<()> {
    cache.clear()?;

    let mut deps = DependencyFile::default();
    for repository in config {
        deps::collect(repository.dir().as_path(), &mut deps);
        index_repository(&cache, &repository)?;
    }

    Ok(())
}

mod metrics {
    pub fn max_line_length(content: &str) -> usize {
        content.lines().map(|x| x.len()).max().unwrap_or(0)
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
            ("solidity", vec!["sol"]),
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

fn tqdm(total: usize) -> Bar {
    tqdm!(total = total, ncols = 40, force_refresh = true)
}
