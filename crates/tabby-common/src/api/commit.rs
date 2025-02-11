use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::Result;

#[async_trait]
pub trait CommitHistorySearch: Send + Sync {
    /// Search git commit history from underlying index.
    ///
    /// * `source_id`: Filter documents by source ID.
    async fn search(
        &self,
        source_id: &str,
        q: &str,
        params: &CommitHistorySearchParams,
    ) -> Result<CommitHistorySearchResponse>;
}

#[derive(Default, Clone, PartialEq, Debug)]
pub struct CommitHistorySearchScores {
    /// Reciprocal rank fusion score: https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html
    pub rrf: f32,
    pub bm25: f32,
    pub embedding: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CommitHistorySearchParams {
    pub min_embedding_score: f32,
    pub min_bm25_score: f32,
    pub min_rrf_score: f32,

    /// At most num_to_return results will be returned.
    pub num_to_return: usize,

    /// At most num_to_score results will be scored.
    pub num_to_score: usize,
}

impl Default for CommitHistorySearchParams {
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

pub struct CommitHistorySearchResponse {
    pub hits: Vec<CommitHistorySearchHit>,
}

#[derive(Clone, Debug)]
pub struct CommitHistorySearchHit {
    pub scores: CommitHistorySearchScores,
    pub commit: CommitHistoryDocument,
}

#[derive(Debug, Clone)]
pub struct CommitHistoryDocument {
    pub git_url: String,
    pub sha: String,
    pub message: String,
    //TODO(kweizh): should we add branches for commit?
    // pub branches: Vec<String>,
    pub author_email: String,
    pub author_at: DateTime<Utc>,
    pub committer: String,
    pub commit_at: DateTime<Utc>,

    pub diff: Option<String>,
}
