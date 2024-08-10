use std::sync::Arc;

use async_trait::async_trait;
use juniper::ID;
use tabby_db::{DbConn, ThreadMessageAttachmentCode};
use tabby_schema::{
    bail,
    thread::{
        self, CreateMessageInput, CreateThreadInput, ThreadRunOptionsInput, ThreadRunStream,
        ThreadService,
    },
    AsID, AsRowid, DbEnum, Result,
};

use super::answer::AnswerService;

struct ThreadServiceImpl {
    db: DbConn,
    answer: Option<Arc<AnswerService>>,
}

#[async_trait]
impl ThreadService for ThreadServiceImpl {
    async fn create(&self, user_id: &ID, input: &CreateThreadInput) -> Result<ID> {
        let thread_id = self.db.create_thread(user_id.as_rowid()?).await?;

        for message in &input.messages {
            let code = message.attachments.as_ref().map(|x| {
                x.code
                    .iter()
                    .map(|x| ThreadMessageAttachmentCode {
                        filepath: x.filepath.clone(),
                        content: x.content.clone(),
                    })
                    .collect::<Vec<_>>()
            });
            self.db
                .create_thread_message(
                    thread_id,
                    message.role.as_enum_str(),
                    &message.content,
                    code.as_deref(),
                    None,
                    false,
                )
                .await?;
        }
        Ok(thread_id.as_id())
    }

    async fn create_run(
        &self,
        thread_id: &ID,
        options: &ThreadRunOptionsInput,
    ) -> Result<ThreadRunStream> {
        let Some(answer) = self.answer.clone() else {
            bail!("Answer service is not available");
        };

        // FIXME(meng): actual lookup messages from database.
        let messages = vec![thread::Message {
            id: ID::new("message:1"),
            thread_id: thread_id.clone(),
            role: thread::Role::User,
            content: "Hello, world!".to_string(),
            attachments: None,
        }];
        answer.answer_v2(&messages, options).await
    }

    async fn append_messages(&self, thread_id: &ID, messages: &[CreateMessageInput]) -> Result<()> {
        let thread_id = thread_id.as_rowid()?;

        for (i, message) in messages.iter().enumerate() {
            let code = message.attachments.as_ref().map(|x| {
                x.code
                    .iter()
                    .map(|x| ThreadMessageAttachmentCode {
                        filepath: x.filepath.clone(),
                        content: x.content.clone(),
                    })
                    .collect::<Vec<_>>()
            });

            let is_first = i == 0;
            self.db
                .create_thread_message(
                    thread_id,
                    message.role.as_enum_str(),
                    &message.content,
                    code.as_deref(),
                    None,
                    // Verify last message role only if it's the first message
                    is_first,
                )
                .await?;
        }

        Ok(())
    }
}

pub fn create(db: DbConn, answer: Option<Arc<AnswerService>>) -> impl ThreadService {
    ThreadServiceImpl { db, answer }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tabby_db::{testutils::create_user, DbConn};
    use tabby_schema::thread::{CreateMessageInput, CreateThreadInput, Role};

    #[tokio::test]
    async fn test_create_thread() {
        let db = DbConn::new_in_memory().await.unwrap();
        let user_id = create_user(&db).await.as_id();
        let service = create(db, None);

        let input = CreateThreadInput {
            messages: vec![CreateMessageInput {
                role: Role::User,
                content: "Hello, world!".to_string(),
                attachments: None,
            }],
        };

        assert!(service.create(&user_id, &input).await.is_ok());
    }

    #[tokio::test]
    async fn test_append_messages() {
        let db = DbConn::new_in_memory().await.unwrap();
        let user_id = create_user(&db).await.as_id();
        let service = create(db, None);

        let thread_id = service
            .create(
                &user_id,
                &CreateThreadInput {
                    messages: vec![CreateMessageInput {
                        role: Role::User,
                        content: "Ping!".to_string(),
                        attachments: None,
                    }],
                },
            )
            .await
            .unwrap();

        assert!(service.append_messages(&thread_id, &vec![
            CreateMessageInput {
                role: Role::User,
                content: "This will not success".to_string(),
                attachments: None,
            }
        ]).await.is_err());

        assert!(service.append_messages(&thread_id, &vec![
            CreateMessageInput {
                role: Role::Assistant,
                content: "Pong!".to_string(),
                attachments: None,
            }
        ]).await.is_ok());
    }
}
