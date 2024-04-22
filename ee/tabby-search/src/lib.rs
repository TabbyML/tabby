use std::path::Path;

use ignore::Walk;

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
        let mut nucleo = nucleo::Matcher::new(nucleo::Config::DEFAULT.match_paths());
        let needle = nucleo::pattern::Pattern::new(
            pattern,
            nucleo::pattern::CaseMatching::Ignore,
            nucleo::pattern::Normalization::Smart,
            nucleo::pattern::AtomKind::Fuzzy,
        );

        let mut scored_entries: Vec<(_, _)> = Walk::new(base)
            // Limit traversal for at most 1M entries for performance reasons.
            .take(1_000_000)
            .filter_map(|path| {
                let entry = path.ok()?;
                let r#type = if entry.file_type().map(|x| x.is_dir()).unwrap_or_default() {
                    "dir".into()
                } else {
                    "file".into()
                };
                let path = entry
                    .into_path()
                    .strip_prefix(base)
                    .ok()?
                    .to_string_lossy()
                    .into_owned();
                let haystack: nucleo::Utf32String = path.clone().into();
                let mut indices = Vec::new();
                let score = needle.indices(haystack.slice(..), &mut nucleo, &mut indices);
                score.map(|score| (score, FileSearch::new(r#type, path, indices)))
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
