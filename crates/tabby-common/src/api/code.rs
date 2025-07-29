use async_trait::async_trait;
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub struct CodeSearchResponse {
    pub hits: Vec<CodeSearchHit>,
}

#[derive(Default, Clone, PartialEq, Debug)]
pub struct CodeSearchHit {
    pub scores: CodeSearchScores,
    pub doc: CodeSearchDocument,
}

#[derive(Default, Clone, PartialEq, Debug)]
pub struct CodeSearchScores {
    /// Reciprocal rank fusion score: https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html
    pub rrf: f32,
    pub bm25: f32,
    pub embedding: f32,
}

#[derive(Builder, Default, Clone, PartialEq, Debug)]
pub struct CodeSearchDocument {
    /// Unique identifier for the file in the repository, stringified SourceFileKey.
    pub file_id: String,
    pub chunk_id: String,

    pub body: String,
    pub filepath: String,
    pub git_url: String,

    // FIXME(kweizh): This should be a required field after 0.25.0.
    // commit represents the specific revision at which the file was last edited.
    pub commit: Option<String>,

    pub language: String,

    /// When start line is `None`, it represents the entire file.
    pub start_line: Option<usize>,
}

#[derive(Error, Debug)]
pub enum CodeSearchError {
    #[error("index not ready")]
    NotReady,

    #[error(transparent)]
    QueryParserError(#[from] tantivy::query::QueryParserError),

    #[error(transparent)]
    TantivyError(#[from] tantivy::TantivyError),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub struct CodeSearchQuery {
    /// filepath in code search query always normalize to unix style.
    pub filepath: Option<String>,
    pub language: Option<String>,
    pub content: String,
    pub source_id: String,
}

impl CodeSearchQuery {
    pub fn new(
        filepath: Option<String>,
        language: Option<String>,
        content: String,
        source_id: String,
    ) -> Self {
        Self {
            filepath: filepath.map(|path| normalize_to_unix_path(&path)),
            language,
            content,
            source_id,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CodeSearchParams {
    pub min_embedding_score: f32,
    pub min_bm25_score: f32,
    pub min_rrf_score: f32,

    /// At most num_to_return results will be returned.
    pub num_to_return: usize,

    /// At most num_to_score results will be scored.
    pub num_to_score: usize,
}

impl Default for CodeSearchParams {
    fn default() -> Self {
        Self {
            min_embedding_score: 0.75,
            min_bm25_score: 8.0,
            min_rrf_score: 0.028,

            num_to_return: 20,
            num_to_score: 40,
        }
    }
}

#[async_trait]
pub trait CodeSearch: Send + Sync {
    async fn search_in_language(
        &self,
        query: CodeSearchQuery,
        params: CodeSearchParams,
    ) -> Result<CodeSearchResponse, CodeSearchError>;
}

/// Normalize the path form different platform to unix style path
pub fn normalize_to_unix_path(filepath: &str) -> String {
    filepath.replace('\\', "/")
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relative_path_normalization() {
        let unix_test_cases = [
            ("./src/main.rs", "./src/main.rs"),
            (".\\src\\main.rs", "./src/main.rs"),
            ("../test/data.json", "../test/data.json"),
            ("..\\test\\data.json", "../test/data.json"),
            ("src/test/file.txt", "src/test/file.txt"),
            ("src\\test\\file.txt", "src/test/file.txt"),
        ];

        for (input, expected) in unix_test_cases {
            assert_eq!(
                normalize_to_unix_path(input),
                expected.to_string(),
                "Failed to normalize path: {input}"
            );
        }
    }
}
