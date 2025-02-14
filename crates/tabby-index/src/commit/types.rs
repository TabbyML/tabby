use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::stream::{BoxStream, StreamExt};
use serde_json::json;

use tabby_common::index::commit::fields;

use crate::indexer::{IndexId, ToIndexId};

pub struct CommitHistory {
    pub source_id: String,

    pub git_url: String,
    pub sha: String,
    pub message: String,
    pub author_email: String,
    pub author_at: DateTime<Utc>,
    pub committer: String,
    pub commit_at: DateTime<Utc>,

    pub diff: Option<String>,
}

impl ToIndexId for CommitHistory {
    fn to_index_id(&self) -> IndexId {
        IndexId {
            source_id: self.source_id.clone(),
            id: format!("{}:::{}", self.git_url, self.sha),
        }
    }
}
