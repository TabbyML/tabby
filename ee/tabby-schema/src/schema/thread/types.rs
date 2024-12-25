use chrono::{DateTime, Utc};
use juniper::{GraphQLEnum, GraphQLInputObject, GraphQLObject, GraphQLUnion, ID};
use serde::Serialize;
use tabby_common::api::{
    code::{CodeSearchDocument, CodeSearchHit, CodeSearchScores},
    structured_doc::DocSearchDocument,
};
use validator::Validate;

use crate::{interface::UserValue, juniper::relay::NodeType, Context};

#[derive(GraphQLEnum, Serialize, Clone, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

#[derive(GraphQLObject, Clone)]
#[graphql(context = Context)]
pub struct Message {
    pub id: ID,
    pub thread_id: ID,
    pub code_source_id: Option<String>,
    pub role: Role,
    pub content: String,

    pub attachment: MessageAttachment,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl NodeType for Message {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "MessageConnection"
    }

    fn edge_type_name() -> &'static str {
        "MessageEdge"
    }
}

#[derive(GraphQLInputObject, Clone, Validate)]
#[graphql(context = Context)]
pub struct UpdateMessageInput {
    pub id: ID,
    pub thread_id: ID,
    #[validate(length(min = 1, code = "content", message = "content can not be empty"))]
    pub content: String,
}

#[derive(GraphQLObject, Clone, Default)]
#[graphql(context = Context)]
pub struct MessageAttachment {
    pub code: Vec<MessageAttachmentCode>,
    pub client_code: Vec<MessageAttachmentClientCode>,
    pub doc: Vec<MessageAttachmentDoc>,
}

#[derive(GraphQLObject, Clone)]
pub struct MessageAttachmentClientCode {
    pub filepath: Option<String>,
    pub start_line: Option<i32>,
    pub content: String,
}

#[derive(GraphQLObject, Clone)]
pub struct MessageAttachmentCode {
    pub git_url: String,
    pub commit: Option<String>,
    pub filepath: String,
    pub language: String,
    pub content: String,

    /// When start line is `None`, it represents the entire file.
    pub start_line: Option<i32>,
}

impl From<CodeSearchDocument> for MessageAttachmentCode {
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
pub struct MessageAttachmentCodeScores {
    pub rrf: f64,
    pub bm25: f64,
    pub embedding: f64,
}

impl From<CodeSearchScores> for MessageAttachmentCodeScores {
    fn from(scores: CodeSearchScores) -> Self {
        Self {
            rrf: scores.rrf as f64,
            bm25: scores.bm25 as f64,
            embedding: scores.embedding as f64,
        }
    }
}

#[derive(GraphQLObject)]
pub struct MessageCodeSearchHit {
    pub code: MessageAttachmentCode,
    pub scores: MessageAttachmentCodeScores,
}

impl From<CodeSearchHit> for MessageCodeSearchHit {
    fn from(hit: CodeSearchHit) -> Self {
        Self {
            code: hit.doc.into(),
            scores: hit.scores.into(),
        }
    }
}

#[derive(GraphQLUnion, Clone)]
#[graphql(context = Context)]
pub enum MessageAttachmentDoc {
    Web(MessageAttachmentWebDoc),
    Issue(MessageAttachmentIssueDoc),
    Pull(MessageAttachmentPullDoc),
}

#[derive(GraphQLObject, Clone)]
pub struct MessageAttachmentWebDoc {
    pub title: String,
    pub link: String,
    pub content: String,
}

#[derive(GraphQLObject, Clone)]
#[graphql(context = Context)]
pub struct MessageAttachmentIssueDoc {
    pub title: String,
    pub link: String,
    pub author: Option<UserValue>,
    pub body: String,
    pub closed: bool,
}

#[derive(GraphQLObject, Clone)]
#[graphql(context = Context)]
pub struct MessageAttachmentPullDoc {
    pub title: String,
    pub link: String,
    pub author: Option<UserValue>,
    pub body: String,
    pub patch: String,
    pub merged: bool,
}

impl MessageAttachmentDoc {
    pub fn from_doc_search_document(doc: DocSearchDocument, author: Option<UserValue>) -> Self {
        match doc {
            DocSearchDocument::Web(web) => MessageAttachmentDoc::Web(MessageAttachmentWebDoc {
                title: web.title,
                link: web.link,
                content: web.snippet,
            }),
            DocSearchDocument::Issue(issue) => {
                MessageAttachmentDoc::Issue(MessageAttachmentIssueDoc {
                    title: issue.title,
                    link: issue.link,
                    author,
                    body: issue.body,
                    closed: issue.closed,
                })
            }
            DocSearchDocument::Pull(pull) => MessageAttachmentDoc::Pull(MessageAttachmentPullDoc {
                title: pull.title,
                link: pull.link,
                author,
                body: pull.body,
                patch: pull.diff,
                merged: pull.merged,
            }),
        }
    }
}

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct MessageDocSearchHit {
    pub doc: MessageAttachmentDoc,
    pub score: f64,
}

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct Thread {
    pub id: ID,
    pub user_id: ID,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl NodeType for Thread {
    type Cursor = String;

    fn cursor(&self) -> Self::Cursor {
        self.id.to_string()
    }

    fn connection_type_name() -> &'static str {
        "ThreadConnection"
    }

    fn edge_type_name() -> &'static str {
        "ThreadEdge"
    }
}

#[derive(GraphQLObject)]
pub struct ThreadCreated {
    pub id: ID,
}

#[derive(GraphQLObject)]
pub struct ThreadRelevantQuestions {
    pub questions: Vec<String>,
}

#[derive(GraphQLObject)]
pub struct ThreadUserMessageCreated {
    pub id: ID,
}

#[derive(GraphQLObject)]
pub struct ThreadAssistantMessageCreated {
    pub id: ID,
}

#[derive(GraphQLObject)]
pub struct ThreadAssistantMessageAttachmentsCode {
    pub code_source_id: String,
    pub hits: Vec<MessageCodeSearchHit>,
}

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct ThreadAssistantMessageAttachmentsDoc {
    pub hits: Vec<MessageDocSearchHit>,
}

#[derive(GraphQLObject)]
pub struct ThreadAssistantMessageContentDelta {
    pub delta: String,
}

#[derive(GraphQLObject)]
pub struct ThreadAssistantMessageCompleted {
    pub id: ID,
}

/// Schema of thread run stream.
///
/// Apart from `thread_message_content_delta`, all other items will only appear once in the stream.
#[derive(GraphQLUnion)]
#[graphql(context = Context)]
pub enum ThreadRunItem {
    ThreadCreated(ThreadCreated),
    ThreadRelevantQuestions(ThreadRelevantQuestions),
    ThreadUserMessageCreated(ThreadUserMessageCreated),
    ThreadAssistantMessageCreated(ThreadAssistantMessageCreated),
    ThreadAssistantMessageAttachmentsCode(ThreadAssistantMessageAttachmentsCode),
    ThreadAssistantMessageAttachmentsDoc(ThreadAssistantMessageAttachmentsDoc),
    ThreadAssistantMessageContentDelta(ThreadAssistantMessageContentDelta),
    ThreadAssistantMessageCompleted(ThreadAssistantMessageCompleted),
}
