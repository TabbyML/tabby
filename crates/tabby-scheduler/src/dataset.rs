use std::{
    collections::HashMap,
    ffi::OsStr,
    fs::{self, read_to_string},
    io::Write,
};

use anyhow::Result;
use file_rotate::{compression::Compression, suffix::AppendCount, ContentLimit, FileRotate};
use lazy_static::lazy_static;
use serde_jsonlines::WriteExt;
use tabby_common::{
    config::{Config, Repository},
    path::dataset_dir,
    SourceFile,
};
use tracing::{error, info};
use tree_sitter_tags::{TagsConfiguration, TagsContext};
use walkdir::{DirEntry, WalkDir};

trait RepositoryExt {
    fn create_dataset(&self, writer: &mut impl Write) -> Result<()>;
}

impl RepositoryExt for Repository {
    fn create_dataset(&self, writer: &mut impl Write) -> Result<()> {
        let dir = self.dir();

        info!("Start indexing repository {}", self.git_url);
        let walk_dir = WalkDir::new(dir.as_path())
            .into_iter()
            .filter_entry(is_not_hidden)
            .filter_map(Result::ok)
            .filter(is_source_code);

        let mut context = TagsContext::new();
        for entry in walk_dir {
            let relative_path = entry.path().strip_prefix(dir.as_path()).unwrap();
            let language = get_language(relative_path.extension().unwrap())
                .unwrap()
                .to_owned();
            if let Ok(file_content) = read_to_string(entry.path()) {
                info!("Building {:?}", relative_path);
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
    let mut writer = FileRotate::new(
        dataset_dir().join("data.jsonl"),
        AppendCount::new(usize::max_value()),
        ContentLimit::Lines(1000),
        Compression::None,
        #[cfg(unix)]
        None,
    );

    for repository in config.repositories.as_slice() {
        repository.create_dataset(&mut writer)?;
    }

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

mod tags {
    use tabby_common::Tag;
    use tree_sitter_tags::TagsContext;

    use super::LANGUAGE_TAGS;

    pub fn collect(context: &mut TagsContext, language: &str, content: &str) -> Vec<Tag> {
        let config = LANGUAGE_TAGS.get(language);
        let empty = Vec::new();

        let Some(config) = config else {
            return empty;
        };

        let Ok((tags, has_error)) = context.generate_tags(&config.0, content.as_bytes(), None)
        else {
            return empty;
        };

        if has_error {
            return empty;
        }

        tags.filter_map(|x| x.ok())
            .map(|x| Tag {
                range: x.range,
                name_range: x.name_range,
                line_range: x.line_range,
                docs: x.docs,
                is_definition: x.is_definition,
                syntax_type_name: config.0.syntax_type_name(x.syntax_type_id).to_owned(),
            })
            .collect()
    }
}

// Mark TagsConfiguration as thread sync / safe.
struct TagsConfigurationSync(TagsConfiguration);
unsafe impl Send for TagsConfigurationSync {}
unsafe impl Sync for TagsConfigurationSync {}

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
    static ref LANGUAGE_TAGS: HashMap<&'static str, TagsConfigurationSync> = {
        HashMap::from([
            (
                "python",
                TagsConfigurationSync(
                    TagsConfiguration::new(
                        tree_sitter_python::language(),
                        tree_sitter_python::TAGGING_QUERY,
                        "",
                    )
                    .unwrap(),
                ),
            ),
            (
                "rust",
                TagsConfigurationSync(
                    TagsConfiguration::new(
                        tree_sitter_rust::language(),
                        tree_sitter_rust::TAGGING_QUERY,
                        "",
                    )
                    .unwrap(),
                ),
            ),
            (
                "javascript-typescript",
                TagsConfigurationSync(
                    TagsConfiguration::new(
                        tree_sitter_typescript::language_tsx(),
                        include_str!("../queries/tsx.scm"),
                        "",
                    )
                    .unwrap(),
                ),
            ),
            (
                "go",
                TagsConfigurationSync(
                    TagsConfiguration::new(
                        tree_sitter_go::language(),
                        include_str!("../queries/go.scm"),
                        "",
                    )
                    .unwrap(),
                ),
            ),
        ])
    };
}
