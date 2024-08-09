use async_trait::async_trait;
use juniper::ID;

use crate::schema::Result;

mod types;
pub use types::*;

mod inputs;
pub use inputs::*;

#[async_trait]
pub trait ThreadService: Send + Sync {
    /// Create a new thread
    async fn create(&self, input: &CreateThreadInput) -> Result<ID>;

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
