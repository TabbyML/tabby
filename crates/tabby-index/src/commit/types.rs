use chrono::{DateTime, TimeZone, Utc};

use crate::indexer::{IndexId, ToIndexId};

pub struct CommitHistory {
    pub source_id: String,

    pub git_url: String,
    pub sha: String,
    pub message: String,
    pub author_email: String,
    pub author_at: DateTime<Utc>,
    pub committer_email: String,
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
