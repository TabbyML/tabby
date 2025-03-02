use juniper::{GraphQLObject, GraphQLUnion};
use tabby_common::api::{
    code::{CodeSearchDocument, CodeSearchScores},
    structured_doc::DocSearchDocument,
};

use crate::{interface::UserValue, Context};

#[derive(GraphQLObject, Clone, Default)]
#[graphql(context = Context)]
/// Represents an attachment to a message, which can include various types of content.
pub struct Attachment {
    /// Code snippets retrieved from the client side.
    pub client_code: Vec<AttachmentClientCode>,

    /// Code snippets retrieved from the server side codebase.
    pub code: Vec<AttachmentCode>,

    /// Documents retrieved from various sources, all from the server side.
    pub doc: Vec<AttachmentDoc>,

    /// File list retrieved from the server side codebase is used for generating this message.
    pub code_file_list: Option<AttachmentCodeFileList>,
}

#[derive(GraphQLObject, Clone)]
pub struct AttachmentCodeFileList {
    pub file_list: Vec<String>,
}

#[derive(GraphQLObject, Clone)]
pub struct AttachmentClientCode {
    pub filepath: Option<String>,
    pub start_line: Option<i32>,
    pub content: String,
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

impl From<CodeSearchDocument> for AttachmentCode {
    fn from(doc: CodeSearchDocument) -> Self {
        Self {
            git_url: doc.git_url,
            commit: doc.commit,
            filepath: doc.filepath,
            language: doc.language,
            content: doc.body,
            start_line: doc.start_line.map(|x| x as i32),
        }
    }
}

#[derive(GraphQLObject, Clone)]
pub struct AttachmentCodeScores {
    pub rrf: f64,
    pub bm25: f64,
    pub embedding: f64,
}

impl From<CodeSearchScores> for AttachmentCodeScores {
    fn from(scores: CodeSearchScores) -> Self {
        Self {
            rrf: scores.rrf as f64,
            bm25: scores.bm25 as f64,
            embedding: scores.embedding as f64,
        }
    }
}

#[derive(GraphQLUnion, Clone)]
#[graphql(context = Context)]
pub enum AttachmentDoc {
    Web(AttachmentWebDoc),
    Issue(AttachmentIssueDoc),
    Pull(AttachmentPullDoc),
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
    pub patch: String,
    pub merged: bool,
}

impl AttachmentDoc {
    pub fn from_doc_search_document(doc: DocSearchDocument, author: Option<UserValue>) -> Self {
        match doc {
            DocSearchDocument::Web(web) => AttachmentDoc::Web(AttachmentWebDoc {
                title: web.title,
                link: web.link,
                content: web.snippet,
            }),
            DocSearchDocument::Issue(issue) => AttachmentDoc::Issue(AttachmentIssueDoc {
                title: issue.title,
                link: issue.link,
                author,
                body: issue.body,
                closed: issue.closed,
            }),
            DocSearchDocument::Pull(pull) => AttachmentDoc::Pull(AttachmentPullDoc {
                title: pull.title,
                link: pull.link,
                author,
                body: pull.body,
                patch: pull.diff,
                merged: pull.merged,
            }),
        }
    }

    pub fn content(&self) -> &str {
        match self {
            AttachmentDoc::Web(web) => &web.content,
            AttachmentDoc::Issue(issue) => &issue.body,
            AttachmentDoc::Pull(pull) => &pull.body,
        }
    }
}

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct MessageDocSearchHit {
    pub doc: AttachmentDoc,
    pub score: f64,
}
