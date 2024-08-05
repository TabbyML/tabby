use async_trait::async_trait;
use juniper::{GraphQLEnum, GraphQLInputObject, GraphQLObject, ID};
use validator::Validate;

use crate::schema::Result;

#[derive(GraphQLEnum)]
pub enum Role {
    User,
    Assistatnt,
}

#[derive(GraphQLInputObject, Validate)]
pub struct CreateMessageInput {
    thread_id: ID,
    role: Role,

    #[validate(length(code = "content", min = 1, message = "Content must not be empty"))]
    content: String,
}

#[derive(GraphQLObject)]
pub struct Message {
    id: ID,
    thread_id: ID,
    role: Role,
    content: String,
}

#[derive(GraphQLInputObject, Validate)]
pub struct CreateThreadMessageInput {
    role: Role,

    #[validate(length(code = "content", min = 1, message = "Content must not be empty"))]
    content: String,
}

#[derive(GraphQLInputObject, Validate)]
pub struct CreateThreadInput {
    messages: Vec<CreateThreadMessageInput>,
}

#[async_trait]
pub trait ThreadService: Send + Sync {
    /// Create a new thread
    async fn create(&self, input: CreateThreadInput) -> Result<ID>;

    /// Delete a thread by ID
    async fn delete(&self, id: ID) -> Result<()>;

    /// Create a new message in a thread
    async fn create_message(&self, input: CreateMessageInput) -> Result<ID>;

    /// Delete a message by ID
    async fn delete_message(&self, id: ID) -> Result<()>;

    /// Query messages in a thread
    async fn list_messages(
        &self,
        thread_id: ID,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<Message>>;
}
