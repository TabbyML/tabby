use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use tantivy::{
    schema::{self, document::CompactDocValue, Value},
    TantivyDocument,
};
use thiserror::Error;

use crate::index::{structured_doc, IndexSchema};

pub struct DocSearchResponse {
    pub hits: Vec<DocSearchHit>,
}

pub struct DocSearchHit {
    pub score: f32,
    pub doc: DocSearchDocument,
}

#[derive(Clone)]
pub enum DocSearchDocument {
    Web(DocSearchWebDocument),
    Issue(DocSearchIssueDocument),
    Pull(DocSearchPullDocument),
    Commit(DocSearchCommit),
    Page(DocSearchPageDocument),
    Ingested(DocSearchIngestedDocument),
}

#[derive(Error, Debug)]
pub enum DocSearchError {
    #[error("index not ready")]
    NotReady,

    #[error(transparent)]
    QueryParserError(#[from] tantivy::query::QueryParserError),

    #[error(transparent)]
    TantivyError(#[from] tantivy::TantivyError),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[async_trait]
pub trait DocSearch: Send + Sync {
    /// Search docs from underlying index.
    ///
    /// * `source_ids`: Filter documents by source IDs, when empty, search all sources.
    async fn search(
        &self,
        source_ids: &[String],
        q: &str,
        limit: usize,
    ) -> Result<DocSearchResponse, DocSearchError>;
}

#[derive(Clone)]
pub struct DocSearchWebDocument {
    pub title: String,
    pub link: String,
    pub snippet: String,
}

#[derive(Clone)]
pub struct DocSearchIssueDocument {
    pub title: String,
    pub link: String,
    pub author_email: Option<String>,
    pub body: String,
    pub closed: bool,
}

#[derive(Clone)]
pub struct DocSearchPullDocument {
    pub title: String,
    pub link: String,
    pub author_email: Option<String>,
    pub body: String,
    pub diff: String,
    pub merged: bool,
}

#[derive(Clone)]
pub struct DocSearchCommit {
    pub sha: String,
    pub message: String,
    pub author_email: String,
    pub author_at: DateTime<Utc>,
}

#[derive(Clone)]
pub struct DocSearchPageDocument {
    pub link: String,
    pub title: String,
    pub content: String,
}

#[derive(Clone)]
pub struct DocSearchIngestedDocument {
    pub id: String,
    pub title: String,
    pub body: String,
    pub link: Option<String>,
}

pub trait FromTantivyDocument {
    fn from_tantivy_document(doc: &TantivyDocument, chunk: &TantivyDocument) -> Option<Self>
    where
        Self: Sized;
}

impl FromTantivyDocument for DocSearchDocument {
    fn from_tantivy_document(doc: &TantivyDocument, chunk: &TantivyDocument) -> Option<Self> {
        let schema = IndexSchema::instance();
        let kind = get_json_text_field(doc, schema.field_attributes, structured_doc::fields::KIND);

        match kind {
            "web" => {
                DocSearchWebDocument::from_tantivy_document(doc, chunk).map(DocSearchDocument::Web)
            }
            "issue" => DocSearchIssueDocument::from_tantivy_document(doc, chunk)
                .map(DocSearchDocument::Issue),
            "pull" => DocSearchPullDocument::from_tantivy_document(doc, chunk)
                .map(DocSearchDocument::Pull),
            "commit" => {
                DocSearchCommit::from_tantivy_document(doc, chunk).map(DocSearchDocument::Commit)
            }
            "page" => DocSearchPageDocument::from_tantivy_document(doc, chunk)
                .map(DocSearchDocument::Page),
            "ingested" => DocSearchIngestedDocument::from_tantivy_document(doc, chunk)
                .map(DocSearchDocument::Ingested),
            _ => None,
        }
    }
}

impl FromTantivyDocument for DocSearchWebDocument {
    fn from_tantivy_document(doc: &TantivyDocument, chunk: &TantivyDocument) -> Option<Self> {
        let schema = IndexSchema::instance();
        let title = get_json_text_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::web::TITLE,
        );
        let link = get_json_text_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::web::LINK,
        );
        let snippet = get_json_text_field(
            chunk,
            schema.field_chunk_attributes,
            structured_doc::fields::web::CHUNK_TEXT,
        );

        Some(Self {
            title: title.into(),
            link: link.into(),
            snippet: snippet.into(),
        })
    }
}

impl FromTantivyDocument for DocSearchIssueDocument {
    fn from_tantivy_document(doc: &TantivyDocument, _: &TantivyDocument) -> Option<Self> {
        let schema = IndexSchema::instance();
        let title = get_json_text_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::issue::TITLE,
        );
        let link = get_json_text_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::issue::LINK,
        );
        let author_email = get_json_option_text_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::issue::AUTHOR_EMAIL,
        );
        let body = get_json_text_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::issue::BODY,
        );
        let closed = get_json_bool_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::issue::CLOSED,
        );
        Some(Self {
            title: title.into(),
            link: link.into(),
            author_email: author_email.map(Into::into),
            body: body.into(),
            closed,
        })
    }
}

impl FromTantivyDocument for DocSearchPullDocument {
    fn from_tantivy_document(doc: &TantivyDocument, _: &TantivyDocument) -> Option<Self> {
        let schema = IndexSchema::instance();
        let title = get_json_text_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::pull::TITLE,
        );
        let link = get_json_text_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::pull::LINK,
        );
        let author_email = get_json_option_text_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::pull::AUTHOR_EMAIL,
        );
        let body = get_json_text_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::pull::BODY,
        );
        let diff = get_json_option_text_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::pull::DIFF,
        );
        let merged = get_json_bool_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::pull::MERGED,
        );
        Some(Self {
            title: title.into(),
            link: link.into(),
            author_email: author_email.map(Into::into),
            body: body.into(),
            diff: diff.unwrap_or_default().into(),
            merged,
        })
    }
}

impl FromTantivyDocument for DocSearchCommit {
    fn from_tantivy_document(doc: &TantivyDocument, _chunk: &TantivyDocument) -> Option<Self> {
        let schema = IndexSchema::instance();
        let sha = get_json_text_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::commit::SHA,
        )
        .to_string();
        let message = get_json_text_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::commit::MESSAGE,
        )
        .to_string();
        let author_email = get_json_text_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::commit::AUTHOR_EMAIL,
        )
        .to_string();
        let author_at = get_json_date_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::commit::AUTHOR_AT,
        )
        .unwrap_or_default();

        Some(Self {
            sha,
            message,
            author_email,
            author_at,
        })
    }
}

impl FromTantivyDocument for DocSearchPageDocument {
    fn from_tantivy_document(doc: &TantivyDocument, chunk: &TantivyDocument) -> Option<Self> {
        let schema = IndexSchema::instance();
        let link = get_json_text_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::page::LINK,
        );
        let title = get_json_text_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::page::TITLE,
        );
        let content = get_json_text_field(
            chunk,
            schema.field_chunk_attributes,
            structured_doc::fields::page::CHUNK_CONTENT,
        );

        Some(Self {
            link: link.into(),
            title: title.into(),
            content: content.into(),
        })
    }
}

impl FromTantivyDocument for DocSearchIngestedDocument {
    fn from_tantivy_document(doc: &TantivyDocument, chunk: &TantivyDocument) -> Option<Self> {
        let schema = IndexSchema::instance();
        let id = doc.get_first(schema.field_id).unwrap().as_str().unwrap();
        let title = get_json_text_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::ingested::TITLE,
        );
        let body = get_json_text_field(
            chunk,
            schema.field_chunk_attributes,
            structured_doc::fields::ingested::CHUNK_BODY,
        );
        let link = get_json_option_text_field(
            doc,
            schema.field_attributes,
            structured_doc::fields::page::LINK,
        );

        Some(Self {
            id: id.into(),
            link: link.map(Into::into),
            title: title.into(),
            body: body.into(),
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

fn get_json_bool_field(doc: &TantivyDocument, field: schema::Field, name: &str) -> bool {
    get_json_field(doc, field, name).as_bool().unwrap()
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

fn get_json_option_text_field<'a>(
    doc: &'a TantivyDocument,
    field: schema::Field,
    name: &str,
) -> Option<&'a str> {
    get_json_option_field(doc, field, name).and_then(|field| field.as_str())
}

fn get_json_date_field(
    doc: &TantivyDocument,
    field: schema::Field,
    name: &str,
) -> Option<DateTime<Utc>> {
    get_json_option_field(doc, field, name)
        .and_then(|field| field.as_datetime())
        .map(|x| x.into_timestamp_secs())
        .and_then(|x| Utc.timestamp_opt(x, 0).single())
}
