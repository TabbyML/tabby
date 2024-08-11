use chrono::{DateTime, Utc};
use derive_builder::Builder;
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
#[derive(GraphQLObject, Builder)]
pub struct ThreadRunItem {
    #[builder(setter(into, strip_option))]
    pub thread_created: Option<ID>,

    #[builder(setter(into, strip_option))]
    pub thread_relevant_questions: Option<Vec<String>>,

    #[builder(setter(into, strip_option))]
    pub thread_user_message_created: Option<ID>,

    #[builder(setter(into, strip_option))]
    pub thread_assistant_message_created: Option<ID>,

    #[builder(setter(into, strip_option))]
    pub thread_assistant_message_attachments_code: Option<Vec<MessageAttachmentCode>>,

    #[builder(setter(into, strip_option))]
    pub thread_assistant_message_attachments_doc: Option<Vec<MessageAttachmentDoc>>,

    #[builder(setter(into, strip_option))]
    pub thread_assistant_message_content_delta: Option<String>,

    #[builder(setter(into, strip_option))]
    pub thread_assistant_message_completed: Option<ID>,
}

impl ThreadRunItem {
    pub fn builder() -> ThreadRunItemBuilder {
        ThreadRunItemBuilder::default()
    }
}

impl ThreadRunItemBuilder {
    pub fn create(&self) -> ThreadRunItem {
        self.build().expect("Failed to build ThreadRunItem")
    }
}