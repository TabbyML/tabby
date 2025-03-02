use chrono::{DateTime, Utc};
use juniper::{GraphQLEnum, GraphQLInputObject, GraphQLObject, GraphQLUnion, ID};
use serde::Serialize;
use tabby_common::api::code::CodeSearchHit;
use validator::Validate;

use super::MessageAttachmentCodeInput;
use crate::{
    juniper::relay::NodeType,
    retrieval::{Attachment, AttachmentCode, AttachmentCodeScores, AttachmentDoc},
    Context,
};

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

    pub attachment: Attachment,

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

#[derive(GraphQLObject, Clone)]
pub struct MessageAttachmentCodeFileList {
    pub file_list: Vec<String>,
}

#[derive(GraphQLObject, Clone)]
pub struct MessageAttachmentClientCode {
    pub filepath: Option<String>,
    pub start_line: Option<i32>,
    pub content: String,
}

impl From<MessageAttachmentClientCode> for MessageAttachmentCodeInput {
    fn from(val: MessageAttachmentClientCode) -> Self {
        MessageAttachmentCodeInput {
            filepath: val.filepath,
            start_line: val.start_line,
            content: val.content,
        }
    }
}

#[derive(GraphQLObject)]
pub struct MessageCodeSearchHit {
    pub code: AttachmentCode,
    pub scores: AttachmentCodeScores,
}

impl From<CodeSearchHit> for MessageCodeSearchHit {
    fn from(hit: CodeSearchHit) -> Self {
        Self {
            code: hit.doc.into(),
            scores: hit.scores.into(),
        }
    }
}

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct MessageDocSearchHit {
    pub doc: AttachmentDoc,
    pub score: f64,
}

#[derive(GraphQLObject)]
#[graphql(context = Context)]
pub struct Thread {
    pub id: ID,
    pub user_id: ID,

    pub is_ephemeral: bool,

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

#[derive(GraphQLObject, Clone, Debug)]
pub struct ThreadAssistantMessageReadingCode {
    pub snippet: bool,
    pub file_list: bool,
    // pub commit_history: bool
}

#[derive(GraphQLObject)]
pub struct ThreadAssistantMessageAttachmentsCodeFileList {
    pub file_list: Vec<String>,
}

#[derive(GraphQLObject)]
pub struct ThreadAssistantMessageAttachmentsCode {
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
    ThreadAssistantMessageReadingCode(ThreadAssistantMessageReadingCode),
    ThreadAssistantMessageAttachmentsCodeFileList(ThreadAssistantMessageAttachmentsCodeFileList),
    ThreadAssistantMessageAttachmentsCode(ThreadAssistantMessageAttachmentsCode),
    ThreadAssistantMessageAttachmentsDoc(ThreadAssistantMessageAttachmentsDoc),
    ThreadAssistantMessageContentDelta(ThreadAssistantMessageContentDelta),
    ThreadAssistantMessageCompleted(ThreadAssistantMessageCompleted),
}
