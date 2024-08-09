use juniper::{GraphQLEnum, GraphQLInputObject, GraphQLObject, ID};
use serde::Serialize;

#[derive(GraphQLEnum, Serialize)]
pub enum Role {
    User,
    Assistant,
}

#[derive(GraphQLObject)]
pub struct Message {
    id: ID,
    thread_id: ID,
    role: Role,
    content: String,

    attachments: Vec<MessageAttachment>,
}

#[derive(GraphQLObject)]
pub struct MessageAttachment {
    code: Vec<MessageAttachmentCode>,
    doc: Vec<MessageAttachmentDoc>,
}

#[derive(GraphQLObject)]
pub struct MessageAttachmentCode {
    pub filepath: Option<String>,
    pub content: String,
}

#[derive(GraphQLObject)]
pub struct MessageAttachmentDoc {
    pub title: String,
    pub link: String,
    pub content: String,
}

#[derive(GraphQLObject)]
pub struct ThreadRunItem {
    thread_created: Option<ID>,
    thread_message_created: Option<ID>,
    thread_message_attachments_code: Vec<MessageAttachmentCode>,
    thread_message_attachments_doc: Vec<MessageAttachmentDoc>,
    thread_message_content_delta: Option<String>,
    thread_message_completed: Option<ID>,
}