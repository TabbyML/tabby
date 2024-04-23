use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

#[derive(Debug)]
pub struct FileSearch {
    pub r#type: String,
    pub path: String,

    /// matched indices for fuzzy search query.
    pub indices: Vec<i32>,
}

impl FileSearch {
    fn new(r#type: String, path: String, indices: Vec<u32>) -> Self {
        Self {
            r#type,
            path,
            indices: indices.into_iter().map(|i| i as i32).collect(),
        }
    }

    pub fn search(
        base: &Path,
        pattern: &str,
        limit: usize,
    ) -> Result<Vec<FileSearch>, anyhow::Error> {
        let paths = {
            let repo = git2::Repository::open(base)?;
            let index = repo.index()?;
            let mut paths = HashSet::new();
            index
                .iter()
                .map(|x| bytes2path(&x.path).to_owned())
                .for_each(|relpath| {
                    let relpath = PathBuf::from(relpath);
                    if let Some(parent) = relpath.parent() {
                        paths.insert(parent.to_owned());
                    };
                    paths.insert(relpath);
                });
            paths.into_iter()
        };

        let mut nucleo = nucleo::Matcher::new(nucleo::Config::DEFAULT.match_paths());
        let needle = nucleo::pattern::Pattern::new(
            pattern,
            nucleo::pattern::CaseMatching::Ignore,
            nucleo::pattern::Normalization::Smart,
            nucleo::pattern::AtomKind::Fuzzy,
        );

        let mut scored_entries: Vec<(_, _)> = paths
            .filter_map(|basepath| {
                let path = PathBuf::from(base).join(&basepath);
                let metadata = path.metadata().ok()?;
                let r#type = if metadata.is_dir() {
                    "dir".into()
                } else {
                    "file".into()
                };
                let basepath = basepath.display().to_string();
                let haystack: nucleo::Utf32String = basepath.clone().into();
                let mut indices = Vec::new();
                let score = needle.indices(haystack.slice(..), &mut nucleo, &mut indices);
                score.map(|score| (score, FileSearch::new(r#type, basepath, indices)))
            })
            // Ensure there's at least 1000 entries with scores > 0 for quality.
            .take(1000)
            .collect();

        scored_entries.sort_by_key(|x| -(x.0 as i32));
        let entries = scored_entries
            .into_iter()
            .map(|x| x.1)
            .take(limit)
            .collect();

        Ok(entries)
    }
}

#[cfg(unix)]
pub fn bytes2path(b: &[u8]) -> &Path {
    use std::os::unix::prelude::*;
    Path::new(std::ffi::OsStr::from_bytes(b))
}
#[cfg(windows)]
pub fn bytes2path(b: &[u8]) -> &Path {
    use std::str;
    Path::new(str::from_utf8(b).unwrap())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::FileSearch;

    #[test]
    fn it_search() {
        let result = FileSearch::search(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../"),
            "website".into(),
            1,
        )
        .unwrap();
        assert_eq!(result.len(), 1);
    }
}
