use std::sync::Arc;

use async_trait::async_trait;
use juniper::ID;
use tabby_schema::{
    thread::{CreateMessageInput, CreateThreadInput, ThreadRunStream, ThreadService},
    Result,
};

use super::answer::AnswerService;

struct ThreadServiceImpl {
    answer: Option<Arc<AnswerService>>,
}

#[async_trait]
impl ThreadService for ThreadServiceImpl {
    async fn create(&self, _input: &CreateThreadInput) -> Result<ID> {
        Ok(ID::new("1"))
    }

    async fn create_run(&self, _id: &ID) -> Result<ThreadRunStream> {
        unimplemented!()
    }

    async fn append_messages(&self, _id: &ID, _messages: &[CreateMessageInput]) -> Result<()> {
        Ok(())
    }
}

pub fn create(answer: Option<Arc<AnswerService>>) -> impl ThreadService {
    ThreadServiceImpl { answer }
}
