use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use tantivy::{
    schema::{self, document::CompactDocValue, Value},
    DateTime as TantivyDateTime, TantivyDocument,
};

use super::Result;
use crate::index::{commit::fields, IndexSchema};

#[async_trait]
pub trait CommitHistorySearch: Send + Sync {
    /// Search git commit history from underlying index.
    ///
    /// * `source_id`: Filter documents by source ID.
    async fn search(
        &self,
        source_id: &str,
        q: &str,
        limit: usize,
    ) -> Result<CommitHistorySearchResponse>;
}

pub struct CommitHistorySearchResponse {
    pub hits: Vec<CommitHistorySearchHit>,
}

#[derive(Clone, Debug)]
pub struct CommitHistorySearchHit {
    pub score: f32,
    pub commit: CommitHistoryDocument,
}

#[derive(Debug, Clone)]
pub struct CommitHistoryDocument {
    pub git_url: String,
    pub sha: String,
    pub message: String,
    pub author_email: String,
    pub author_at: DateTime<Utc>,
    pub committer: String,
    pub commit_at: DateTime<Utc>,

    pub diff: Option<String>,
    pub changed_file: Option<String>,
}

impl CommitHistoryDocument {
    pub fn from_tantivy_document(doc: &TantivyDocument, chunk: &TantivyDocument) -> Option<Self> {
        let schema = IndexSchema::instance();
        let git_url =
            get_json_text_field(doc, schema.field_attributes, fields::GIT_URL).to_string();
        let sha = get_json_text_field(doc, schema.field_attributes, fields::SHA).to_string();
        let message =
            get_json_text_field(doc, schema.field_attributes, fields::MESSAGE).to_string();
        let author_email =
            get_json_text_field(doc, schema.field_attributes, fields::AUTHOR_EMAIL).to_string();
        let author_at = get_json_date_field(doc, schema.field_attributes, fields::AUTHOR_AT)
            .unwrap()
            .into_timestamp_secs();
        let committer =
            get_json_text_field(doc, schema.field_attributes, fields::COMMITTER).to_string();
        let commit_at = get_json_date_field(doc, schema.field_attributes, fields::COMMIT_AT)
            .unwrap()
            .into_timestamp_secs();
        let diff =
            get_json_option_text_field(chunk, schema.field_chunk_attributes, fields::CHUNK_DIFF)
                .map(|s| s.to_string());
        let changed_file = get_json_option_text_field(
            chunk,
            schema.field_chunk_attributes,
            fields::CHUNK_FILEPATH,
        )
        .map(|s| s.to_string());

        Some(Self {
            git_url,
            sha,
            message,
            author_email,
            author_at: Utc.timestamp_opt(author_at, 0).single().unwrap_or_default(),
            committer,
            commit_at: Utc.timestamp_opt(commit_at, 0).single().unwrap_or_default(),

            diff,
            changed_file,
        })
    }
}

fn get_json_field<'a>(
    doc: &'a TantivyDocument,
    field: schema::Field,
    name: &str,
) -> CompactDocValue<'a> {
    doc.get_first(field)
        .unwrap()
        .as_object()
        .unwrap()
        .find(|(k, _)| *k == name)
        .unwrap()
        .1
}

fn get_json_text_field<'a>(doc: &'a TantivyDocument, field: schema::Field, name: &str) -> &'a str {
    get_json_field(doc, field, name).as_str().unwrap()
}

fn get_json_option_field<'a>(
    doc: &'a TantivyDocument,
    field: schema::Field,
    name: &str,
) -> Option<CompactDocValue<'a>> {
    Some(
        doc.get_first(field)?
            .as_object()?
            .find(|(k, _)| *k == name)?
            .1,
    )
}

fn get_json_date_field(
    doc: &TantivyDocument,
    field: schema::Field,
    name: &str,
) -> Option<TantivyDateTime> {
    get_json_option_field(doc, field, name).and_then(|field| field.as_datetime())
}

fn get_json_option_text_field<'a>(
    doc: &'a TantivyDocument,
    field: schema::Field,
    name: &str,
) -> Option<&'a str> {
    get_json_option_field(doc, field, name).and_then(|field| field.as_str())
}
