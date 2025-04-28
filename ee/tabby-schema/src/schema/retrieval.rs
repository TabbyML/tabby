use chrono::{DateTime, Utc};
use juniper::{GraphQLObject, GraphQLUnion};
use tabby_common::api::code::{CodeSearchDocument, CodeSearchHit, CodeSearchScores};

use crate::{
    interface::UserValue,
    thread::{
        MessageAttachmentCode, MessageAttachmentCodeFileList, MessageAttachmentCommitDoc,
        MessageAttachmentDoc, MessageAttachmentIngestedDoc, MessageAttachmentIssueDoc,
        MessageAttachmentPageDoc, MessageAttachmentPullDoc, MessageAttachmentWebDoc,
    },
    Context,
};

#[derive(GraphQLObject)]
pub struct AttachmentCodeHits {
    pub hits: Vec<AttachmentCodeHit>,
}

#[derive(GraphQLObject)]
pub struct AttachmentCodeHit {
    pub code: AttachmentCode,
    pub scores: AttachmentCodeScores,
}

#[derive(GraphQLObject, Clone)]
pub struct AttachmentCodeScores {
    pub rrf: f64,
    pub bm25: f64,
    pub embedding: f64,
}

#[derive(GraphQLObject, Clone, PartialEq)]
pub struct AttachmentCode {
    pub git_url: String,
    pub commit: Option<String>,
    pub filepath: String,
    pub language: String,
    pub content: String,

    /// When start line is `None`, it represents the entire file.
    pub start_line: Option<i32>,
}

#[derive(GraphQLObject, Clone)]
pub struct AttachmentCodeFileList {
    pub file_list: Vec<String>,
    pub truncated: bool,
}

#[derive(GraphQLObject, Clone)]
#[graphql(context = Context)]
pub struct AttachmentDocHit {
    pub doc: AttachmentDoc,
    pub score: f64,
}

#[derive(GraphQLUnion, Clone)]
#[graphql(context = Context)]
pub enum AttachmentDoc {
    Web(AttachmentWebDoc),
    Issue(AttachmentIssueDoc),
    Pull(AttachmentPullDoc),
    Commit(AttachmentCommitDoc),
    Page(AttachmentPageDoc),
    Ingested(AttachmentIngestedDoc),
}

impl PartialEq for AttachmentDoc {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (AttachmentDoc::Web(a), AttachmentDoc::Web(b)) => a.link == b.link,
            (AttachmentDoc::Issue(a), AttachmentDoc::Issue(b)) => a.link == b.link,
            (AttachmentDoc::Pull(a), AttachmentDoc::Pull(b)) => a.link == b.link,
            (AttachmentDoc::Commit(a), AttachmentDoc::Commit(b)) => a.sha == b.sha,
            (AttachmentDoc::Page(a), AttachmentDoc::Page(b)) => a.link == b.link,
            (AttachmentDoc::Ingested(a), AttachmentDoc::Ingested(b)) => {
                a.id == b.id && a.title == b.title
            }
            _ => false,
        }
    }
}

#[derive(GraphQLObject, Clone)]
pub struct AttachmentWebDoc {
    pub title: String,
    pub link: String,
    pub content: String,
}

#[derive(GraphQLObject, Clone)]
#[graphql(context = Context)]
pub struct AttachmentIssueDoc {
    pub title: String,
    pub link: String,
    pub author: Option<UserValue>,
    pub body: String,
    pub closed: bool,
}

#[derive(GraphQLObject, Clone)]
#[graphql(context = Context)]
pub struct AttachmentPullDoc {
    pub title: String,
    pub link: String,
    pub author: Option<UserValue>,
    pub body: String,
    pub diff: String,
    pub merged: bool,
}

#[derive(GraphQLObject, Clone)]
#[graphql(context = Context)]
pub struct AttachmentCommitDoc {
    pub sha: String,
    pub message: String,
    pub author: Option<UserValue>,
    pub author_at: DateTime<Utc>,
}

#[derive(GraphQLObject, Clone)]
#[graphql(context = Context)]
pub struct AttachmentPageDoc {
    pub link: String,
    pub title: String,
    pub content: String,
}

#[derive(GraphQLObject, Clone)]
#[graphql(context = Context)]
pub struct AttachmentIngestedDoc {
    pub id: String,
    pub title: String,
    pub body: String,
    pub link: Option<String>,
}

impl From<&MessageAttachmentCodeFileList> for AttachmentCodeFileList {
    fn from(value: &MessageAttachmentCodeFileList) -> Self {
        Self {
            file_list: value.file_list.clone(),
            truncated: value.truncated,
        }
    }
}

impl From<AttachmentCodeFileList> for MessageAttachmentCodeFileList {
    fn from(value: AttachmentCodeFileList) -> Self {
        Self {
            file_list: value.file_list,
            truncated: value.truncated,
        }
    }
}

impl From<&MessageAttachmentCode> for AttachmentCode {
    fn from(value: &MessageAttachmentCode) -> Self {
        Self {
            git_url: value.git_url.clone(),
            commit: value.commit.clone(),
            filepath: value.filepath.clone(),
            language: value.language.clone(),
            content: value.content.clone(),
            start_line: value.start_line,
        }
    }
}

impl From<&AttachmentCode> for MessageAttachmentCode {
    fn from(value: &AttachmentCode) -> Self {
        Self {
            git_url: value.git_url.clone(),
            commit: value.commit.clone(),
            filepath: value.filepath.clone(),
            language: value.language.clone(),
            content: value.content.clone(),
            start_line: value.start_line,
        }
    }
}

impl From<CodeSearchHit> for AttachmentCodeHit {
    fn from(val: CodeSearchHit) -> Self {
        Self {
            code: val.doc.into(),
            scores: val.scores.into(),
        }
    }
}

impl From<CodeSearchDocument> for AttachmentCode {
    fn from(val: CodeSearchDocument) -> Self {
        Self {
            git_url: val.git_url.clone(),
            commit: val.commit.clone(),
            filepath: val.filepath.clone(),
            language: val.language.clone(),
            content: val.body.clone(),
            start_line: val.start_line.map(|x| x as i32),
        }
    }
}

impl From<CodeSearchScores> for AttachmentCodeScores {
    fn from(val: CodeSearchScores) -> Self {
        Self {
            rrf: val.rrf as f64,
            bm25: val.bm25 as f64,
            embedding: val.embedding as f64,
        }
    }
}

impl From<&MessageAttachmentDoc> for AttachmentDoc {
    fn from(val: &MessageAttachmentDoc) -> Self {
        match val {
            MessageAttachmentDoc::Web(doc) => AttachmentDoc::Web(doc.into()),
            MessageAttachmentDoc::Issue(doc) => AttachmentDoc::Issue(doc.into()),
            MessageAttachmentDoc::Pull(doc) => AttachmentDoc::Pull(doc.into()),
            MessageAttachmentDoc::Commit(doc) => AttachmentDoc::Commit(doc.into()),
            MessageAttachmentDoc::Page(doc) => AttachmentDoc::Page(doc.into()),
            MessageAttachmentDoc::Ingested(doc) => AttachmentDoc::Ingested(doc.into()),
        }
    }
}

impl From<&MessageAttachmentWebDoc> for AttachmentWebDoc {
    fn from(val: &MessageAttachmentWebDoc) -> Self {
        Self {
            title: val.title.clone(),
            link: val.link.clone(),
            content: val.content.clone(),
        }
    }
}

impl From<&MessageAttachmentIssueDoc> for AttachmentIssueDoc {
    fn from(val: &MessageAttachmentIssueDoc) -> Self {
        Self {
            title: val.title.clone(),
            link: val.link.clone(),
            author: val.author.clone(),
            body: val.body.clone(),
            closed: val.closed,
        }
    }
}

impl From<&MessageAttachmentPullDoc> for AttachmentPullDoc {
    fn from(val: &MessageAttachmentPullDoc) -> Self {
        Self {
            title: val.title.clone(),
            link: val.link.clone(),
            author: val.author.clone(),
            body: val.body.clone(),
            diff: val.patch.clone(),
            merged: val.merged,
        }
    }
}

impl From<&MessageAttachmentCommitDoc> for AttachmentCommitDoc {
    fn from(val: &MessageAttachmentCommitDoc) -> Self {
        Self {
            sha: val.sha.clone(),
            message: val.message.clone(),
            author: val.author.clone(),
            author_at: val.author_at,
        }
    }
}

impl From<&MessageAttachmentPageDoc> for AttachmentPageDoc {
    fn from(val: &MessageAttachmentPageDoc) -> Self {
        Self {
            link: val.link.clone(),
            title: val.title.clone(),
            content: val.content.clone(),
        }
    }
}

impl From<&MessageAttachmentIngestedDoc> for AttachmentIngestedDoc {
    fn from(val: &MessageAttachmentIngestedDoc) -> Self {
        Self {
            id: val.id.clone(),
            title: val.title.clone(),
            body: val.body.clone(),
            link: val.link.clone(),
        }
    }
}
