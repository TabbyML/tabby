use async_trait::async_trait;
use futures::stream::BoxStream;
use juniper::ID;

use crate::{policy::AccessPolicy, schema::Result};

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

    /// Delete a thread.
    async fn delete(&self, id: &ID) -> Result<()>;

    /// Converting a ephemeral thread to a persisted thread
    async fn set_persisted(&self, id: &ID) -> Result<()>;

    /// List threads
    async fn list(
        &self,
        ids: Option<&[ID]>,
        is_ephemeral: Option<bool>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<Thread>>;

    /// List threads owned by a user
    async fn list_owned(
        &self,
        user_id: &ID,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<Thread>>;

    /// Create a new thread run
    async fn create_run(
        &self,
        policy: &AccessPolicy,
        id: &ID,
        options: &ThreadRunOptionsInput,
        attachment_input: Option<&MessageAttachmentInput>,
        yield_last_user_message: bool,
        yield_thread_created: bool,
    ) -> Result<ThreadRunStream>;

    /// Append message to an existing thread
    async fn append_user_message(&self, id: &ID, message: &CreateMessageInput) -> Result<()>;

    /// Update a message
    async fn update_thread_message(&self, message: &UpdateMessageInput) -> Result<()>;

    // /// Delete a thread by ID
    // async fn delete(&self, id: ID) -> Result<()>;

    /// Query messages in a thread
    async fn list_thread_messages(
        &self,
        thread_id: &ID,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<Message>>;

    /// Delete pair of user message and bot response in a thread.
    async fn delete_thread_message_pair(
        &self,
        thread_id: &ID,
        user_message_id: &ID,
        assistant_message_id: &ID,
    ) -> Result<()>;
}
