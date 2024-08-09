use std::sync::Arc;

use async_trait::async_trait;
use juniper::ID;
use tabby_schema::{
    bail,
    thread::{
        self, CreateMessageInput, CreateThreadInput, ThreadRunOptionsInput, ThreadRunStream,
        ThreadService,
    },
    Result,
};

use super::answer::AnswerService;

struct ThreadServiceImpl {
    answer: Option<Arc<AnswerService>>,
}

#[async_trait]
impl ThreadService for ThreadServiceImpl {
    async fn create(&self, _input: &CreateThreadInput) -> Result<ID> {
        Ok(ID::new("message:1"))
    }

    async fn create_run(
        &self,
        _id: &ID,
        options: &ThreadRunOptionsInput,
    ) -> Result<ThreadRunStream> {
        let Some(answer) = self.answer.clone() else {
            bail!("Answer service is not available");
        };

        // FIXME(meng): actual lookup messages from database.
        let messages = vec![thread::Message {
            id: ID::new("message:1"),
            thread_id: ID::new("thread:1"),
            role: thread::Role::User,
            content: "Hello, world!".to_string(),
            attachments: None,
        }];
        answer.answer_v2(&messages, options).await
    }

    async fn append_messages(&self, _id: &ID, _messages: &[CreateMessageInput]) -> Result<()> {
        Ok(())
    }
}

pub fn create(answer: Option<Arc<AnswerService>>) -> impl ThreadService {
    ThreadServiceImpl { answer }
}
