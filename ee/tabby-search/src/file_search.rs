use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use git2::TreeWalkResult;

pub struct GitFileSearch {
    pub r#type: &'static str,
    pub path: String,

    /// matched indices for fuzzy search query.
    pub indices: Vec<i32>,
}

impl GitFileSearch {
    fn new(r#type: &'static str, path: String, indices: Vec<u32>) -> Self {
        Self {
            r#type,
            path,
            indices: indices.into_iter().map(|i| i as i32).collect(),
        }
    }

    pub fn search(
        repository: &git2::Repository,
        commit: Option<&str>,
        pattern: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<GitFileSearch>> {
        let paths = {
            let commit = if let Some(commit) = commit {
                repository.find_commit(git2::Oid::from_str(commit)?)?
            } else {
                repository.head()?.peel_to_commit()?
            };

            let tree = commit.tree()?;
            let mut paths = vec![];
            tree.walk(git2::TreeWalkMode::PreOrder, |path, entry| {
                let path = PathBuf::from(path).join(bytes2path(entry.name_bytes()));
                paths.push((entry.kind() == Some(git2::ObjectType::Blob), path));
                TreeWalkResult::Ok
            })?;

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
            .filter_map(|(is_file, basepath)| {
                let r#type = if is_file { "file" } else { "dir" };
                let basepath = basepath.display().to_string();
                let haystack: nucleo::Utf32String = basepath.clone().into();
                let mut indices = Vec::new();
                let score = needle.indices(haystack.slice(..), &mut nucleo, &mut indices);
                score.map(|score| (score, GitFileSearch::new(r#type, basepath, indices)))
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
    use crate::{testutils::TempGitRepository, GitFileSearch};

    #[test]
    fn it_search() {
        let root = TempGitRepository::default();

        let result =
            GitFileSearch::search(&root.repository(), None, "moonscript_lora md", 5).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].r#type, "file");
        assert_eq!(result[0].path, "201_lm_moonscript_lora/README.md");
    }
}
