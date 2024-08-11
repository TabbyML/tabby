use async_trait::async_trait;
use futures::stream::BoxStream;
use juniper::ID;

use crate::schema::Result;

mod types;
pub use types::*;

mod inputs;
pub use inputs::*;

pub type ThreadRunStream = BoxStream<'static, Result<ThreadRunItem>>;

#[async_trait]
pub trait ThreadService: Send + Sync {
    /// Create a new thread
    async fn create(&self, user_id: &ID, input: &CreateThreadInput) -> Result<ID>;

    /// Get a thread by ID
    async fn get(&self, id: &ID) -> Result<Option<Thread>>;

    /// Create a new thread run
    async fn create_run(
        &self,
        id: &ID,
        options: &ThreadRunOptionsInput,
        yield_last_user_message: bool,
        yield_thread_created: bool,
    ) -> Result<ThreadRunStream>;

    /// Append message to an existing thread
    async fn append_user_message(&self, id: &ID, message: &CreateMessageInput) -> Result<()>;

    // /// Delete a thread by ID
    // async fn delete(&self, id: ID) -> Result<()>;

    // /// Create a new message in a thread
    // async fn create_message(&self, input: CreateMessageInput) -> Result<ID>;

    // /// Delete a message by ID
    // async fn delete_message(&self, id: ID) -> Result<()>;

    // /// Query messages in a thread
    // async fn list_messages(
    //     &self,
    //     thread_id: ID,
    //     after: Option<String>,
    //     before: Option<String>,
    //     first: Option<usize>,
    //     last: Option<usize>,
    // ) -> Result<Vec<Message>>;
}
