use std::sync::Arc;

use juniper::ID;
use tabby_schema::thread::{CreateMessageInput, CreateThreadInput, ThreadRunStream, ThreadService};
use async_trait::async_trait;
use tabby_schema::Result;

use super::answer::AnswerService;

struct ThreadServiceImpl {
    answer: Option<Arc<AnswerService>>,
}

#[async_trait]
impl ThreadService for ThreadServiceImpl {
    async fn create(&self, input: &CreateThreadInput) -> Result<ID> {
        Ok(ID::new("1"))
    }

    async fn create_run(&self, id: &ID) -> Result<ThreadRunStream> {
        unimplemented!()
    }

    async fn append_messages(&self, id: &ID, messages: &[CreateMessageInput]) -> Result<()> {
        Ok(())
    }
}

pub fn create(answer: Option<Arc<AnswerService>>) -> impl ThreadService {
    ThreadServiceImpl {
        answer
    }
}