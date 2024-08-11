use chrono::{DateTime, Utc};
use juniper::{GraphQLEnum, GraphQLObject, ID};
use serde::Serialize;

#[derive(GraphQLEnum, Serialize, Clone, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

#[derive(GraphQLObject, Clone)]
pub struct Message {
    pub id: ID,
    pub thread_id: ID,
    pub role: Role,
    pub content: String,

    pub attachment: MessageAttachment,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
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
pub struct Thread {
    pub id: ID,
    pub user_id: ID,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>
}

/// Schema of thread run stream.
///
/// Apart from `thread_message_content_delta`, all other items will only appear once in the stream.
#[derive(GraphQLObject)]
pub struct ThreadRunItem {
    pub thread_created: Option<ID>,
    pub thread_message_created: Option<ID>,
    pub thread_message_attachments_code: Option<Vec<MessageAttachmentCode>>,
    pub thread_message_attachments_doc: Option<Vec<MessageAttachmentDoc>>,
    pub thread_relevant_questions: Option<Vec<String>>,
    pub thread_message_content_delta: Option<String>,
    pub thread_message_completed: Option<ID>,
}

impl ThreadRunItem {
    pub fn thread_created(id: ID) -> Self {
        Self {
            thread_created: Some(id),
            thread_message_created: None,
            thread_message_attachments_code: None,
            thread_message_attachments_doc: None,
            thread_relevant_questions: None,
            thread_message_content_delta: None,
            thread_message_completed: None,
        }
    }

    pub fn thread_message_created(id: ID) -> Self {
        Self {
            thread_created: None,
            thread_message_created: Some(id),
            thread_message_attachments_code: None,
            thread_message_attachments_doc: None,
            thread_relevant_questions: None,
            thread_message_content_delta: None,
            thread_message_completed: None,
        }
    }

    pub fn thread_message_completed(id: ID) -> Self {
        Self {
            thread_created: None,
            thread_message_created: None,
            thread_message_attachments_code: None,
            thread_message_attachments_doc: None,
            thread_relevant_questions: None,
            thread_message_content_delta: None,
            thread_message_completed: Some(id),
        }
    }

    pub fn thread_message_attachments_code(code: Vec<MessageAttachmentCode>) -> Self {
        Self {
            thread_created: None,
            thread_message_created: None,
            thread_message_attachments_code: Some(code),
            thread_message_attachments_doc: None,
            thread_relevant_questions: None,
            thread_message_content_delta: None,
            thread_message_completed: None,
        }
    }

    pub fn thread_relevant_questions(questions: Vec<String>) -> Self {
        Self {
            thread_created: None,
            thread_message_created: None,
            thread_message_attachments_code: None,
            thread_message_attachments_doc: None,
            thread_relevant_questions: Some(questions),
            thread_message_content_delta: None,
            thread_message_completed: None,
        }
    }

    pub fn thread_message_attachments_doc(doc: Vec<MessageAttachmentDoc>) -> Self {
        Self {
            thread_created: None,
            thread_message_created: None,
            thread_message_attachments_code: None,
            thread_message_attachments_doc: Some(doc),
            thread_relevant_questions: None,
            thread_message_content_delta: None,
            thread_message_completed: None,
        }
    }

    pub fn thread_message_content_delta(delta: String) -> Self {
        Self {
            thread_created: None,
            thread_message_created: None,
            thread_message_attachments_code: None,
            thread_message_attachments_doc: None,
            thread_relevant_questions: None,
            thread_message_content_delta: Some(delta),
            thread_message_completed: None,
        }
    }
}
