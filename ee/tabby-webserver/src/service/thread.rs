use std::sync::Arc;

use async_trait::async_trait;
use juniper::ID;
use tabby_schema::{
    bail,
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
        let Some(_answer) = self.answer.as_ref() else {
            bail!("Answer service is not available");
        };

        todo!("Create a stream of thread run items");
    }

    async fn append_messages(&self, _id: &ID, _messages: &[CreateMessageInput]) -> Result<()> {
        Ok(())
    }
}

pub fn create(answer: Option<Arc<AnswerService>>) -> impl ThreadService {
    ThreadServiceImpl { answer }
}
