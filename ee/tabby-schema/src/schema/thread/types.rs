use juniper::{GraphQLEnum, GraphQLObject, ID};
use serde::Serialize;

#[derive(GraphQLEnum, Serialize, Clone)]
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

    pub attachments: Option<MessageAttachment>,
}

#[derive(GraphQLObject, Clone)]
pub struct MessageAttachment {
    pub code: Vec<MessageAttachmentCode>,
    pub doc: Vec<MessageAttachmentDoc>,
}

/// WARNING: this struct's json representation will be serialized into DB as an individual column (array of objects).
/// If you want to change the struct, please make sure the change is backward compatible.
#[derive(GraphQLObject, Clone)]
pub struct MessageAttachmentCode {
    pub filepath: Option<String>,
    pub content: String,
}

/// WARNING: this struct's json representation will be serialized into DB as an individual column (array of objects).
/// If you want to change the struct, please make sure the change is backward compatible.
#[derive(GraphQLObject, Clone)]
pub struct MessageAttachmentDoc {
    pub title: String,
    pub link: String,
    pub content: String,
}

/// Schema of thread run stream.
///
/// The event's order is kept as same as the order defined in the struct fields.
/// Apart from `thread_message_content_delta`, all other items will only appear once in the stream.
#[derive(GraphQLObject)]
pub struct ThreadRunItem {
    thread_created: Option<ID>,
    thread_message_created: Option<ID>,
    thread_message_attachments_code: Option<Vec<MessageAttachmentCode>>,
    thread_message_attachments_doc: Option<Vec<MessageAttachmentDoc>>,
    thread_message_relevant_questions: Option<Vec<String>>,
    thread_message_content_delta: Option<String>,
    thread_message_completed: Option<ID>,
}

impl ThreadRunItem {
    pub fn thread_message_attachments_code(code: Vec<MessageAttachmentCode>) -> Self {
        Self {
            thread_created: None,
            thread_message_created: None,
            thread_message_attachments_code: Some(code),
            thread_message_attachments_doc: None,
            thread_message_relevant_questions: None,
            thread_message_content_delta: None,
            thread_message_completed: None,
        }
    }

    pub fn thread_message_relevant_questions(questions: Vec<String>) -> Self {
        Self {
            thread_created: None,
            thread_message_created: None,
            thread_message_attachments_code: None,
            thread_message_attachments_doc: None,
            thread_message_relevant_questions: Some(questions),
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
            thread_message_relevant_questions: None,
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
            thread_message_relevant_questions: None,
            thread_message_content_delta: Some(delta),
            thread_message_completed: None,
        }
    }
}
