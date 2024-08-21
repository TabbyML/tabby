use chrono::{DateTime, Utc};
use juniper::{graphql_object, GraphQLEnum, GraphQLObject, ID};
use serde::Serialize;
use tabby_common::api::{
    code::{CodeSearchDocument, CodeSearchHit, CodeSearchScores},
    doc::{DocSearchDocument, DocSearchHit},
};

use crate::{juniper::relay::NodeType, Context};

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

#[derive(GraphQLObject, Clone, Default)]
pub struct MessageAttachment {
    pub code: Vec<MessageAttachmentCode>,
    pub client_code: Vec<MessageAttachmentClientCode>,
    pub doc: Vec<MessageAttachmentDoc>,
}

#[derive(GraphQLObject, Clone)]
pub struct MessageAttachmentClientCode {
    pub filepath: Option<String>,
    pub content: String,
}

#[derive(GraphQLObject, Clone)]
pub struct MessageAttachmentCode {
    pub git_url: String,
    pub filepath: String,
    pub language: String,
    pub content: String,
    pub start_line: i32,
}

impl From<CodeSearchDocument> for MessageAttachmentCode {
    fn from(doc: CodeSearchDocument) -> Self {
        Self {
            git_url: doc.git_url,
            filepath: doc.filepath,
            language: doc.language,
            content: doc.body,
            start_line: doc.start_line as i32,
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

#[derive(GraphQLObject, Clone)]
pub struct MessageAttachmentDoc {
    pub title: String,
    pub link: String,
    pub content: String,
}

impl From<DocSearchDocument> for MessageAttachmentDoc {
    fn from(doc: DocSearchDocument) -> Self {
        Self {
            title: doc.title,
            link: doc.link,
            content: doc.snippet,
        }
    }
}

#[derive(GraphQLObject)]
pub struct MessageDocSearchHit {
    pub doc: MessageAttachmentDoc,
    pub score: f64,
}

impl From<DocSearchHit> for MessageDocSearchHit {
    fn from(hit: DocSearchHit) -> Self {
        Self {
            doc: hit.doc.into(),
            score: hit.score as f64,
        }
    }
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

/// Schema of thread run stream.
///
/// Apart from `thread_message_content_delta`, all other items will only appear once in the stream.
pub enum ThreadRunItem {
    ThreadCreated(ID),
    ThreadRelevantQuestions(Vec<String>),
    ThreadUserMessageCreated(ID),
    ThreadAssistantMessageCreated(ID),
    ThreadAssistantMessageAttachmentsCode(Vec<MessageCodeSearchHit>),
    ThreadAssistantMessageAttachmentsDoc(Vec<MessageDocSearchHit>),
    ThreadAssistantMessageContentDelta(String),
    ThreadAssistantMessageCompleted(ID),
}

#[graphql_object]
impl ThreadRunItem {
    fn thread_created(&self) -> Option<&ID> {
        match self {
            ThreadRunItem::ThreadCreated(id) => Some(id),
            _ => None,
        }
    }

    fn thread_relevant_questions(&self) -> Option<&Vec<String>> {
        match self {
            ThreadRunItem::ThreadRelevantQuestions(questions) => Some(questions),
            _ => None,
        }
    }

    fn thread_user_message_created(&self) -> Option<&ID> {
        match self {
            ThreadRunItem::ThreadUserMessageCreated(id) => Some(id),
            _ => None,
        }
    }

    fn thread_assistant_message_created(&self) -> Option<&ID> {
        match self {
            ThreadRunItem::ThreadAssistantMessageCreated(id) => Some(id),
            _ => None,
        }
    }

    fn thread_assistant_message_attachments_code(&self) -> Option<&Vec<MessageCodeSearchHit>> {
        match self {
            ThreadRunItem::ThreadAssistantMessageAttachmentsCode(hits) => Some(hits),
            _ => None,
        }
    }

    fn thread_assistant_message_attachments_doc(&self) -> Option<&Vec<MessageDocSearchHit>> {
        match self {
            ThreadRunItem::ThreadAssistantMessageAttachmentsDoc(hits) => Some(hits),
            _ => None,
        }
    }

    fn thread_assistant_message_content_delta(&self) -> Option<&String> {
        match self {
            ThreadRunItem::ThreadAssistantMessageContentDelta(content) => Some(content),
            _ => None,
        }
    }

    fn thread_assistant_message_completed(&self) -> Option<&ID> {
        match self {
            ThreadRunItem::ThreadAssistantMessageCompleted(id) => Some(id),
            _ => None,
        }
    }
}
