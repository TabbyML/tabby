use chrono::{DateTime, Utc};
use juniper::{graphql_object, GraphQLEnum, GraphQLObject, ID};
use serde::Serialize;

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
    pub doc: Vec<MessageAttachmentDoc>,
}

/// If you want to change the struct, please make sure the change is backward compatible.
#[derive(GraphQLObject, Clone)]
pub struct MessageAttachmentCode {
    pub filepath: Option<String>,
    pub content: String,
}

#[derive(GraphQLObject, Clone)]
pub struct MessageAttachmentDoc {
    pub title: String,
    pub link: String,
    pub content: String,
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
    ThreadAssistantMessageAttachmentsCode(Vec<MessageAttachmentCode>),
    ThreadAssistantMessageAttachmentsDoc(Vec<MessageAttachmentDoc>),
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

    fn thread_assistant_message_attachments_code(&self) -> Option<&Vec<MessageAttachmentCode>> {
        match self {
            ThreadRunItem::ThreadAssistantMessageAttachmentsCode(attachments) => Some(attachments),
            _ => None,
        }
    }

    fn thread_assistant_message_attachments_doc(&self) -> Option<&Vec<MessageAttachmentDoc>> {
        match self {
            ThreadRunItem::ThreadAssistantMessageAttachmentsDoc(attachments) => Some(attachments),
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
