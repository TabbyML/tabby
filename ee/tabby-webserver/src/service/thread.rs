use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use juniper::ID;
use tabby_db::{DbConn, ThreadMessageAttachmentCode};
use tabby_schema::{
    bail,
    thread::{
        self, CreateMessageInput, CreateThreadInput, MessageAttachment, ThreadRunItem,
        ThreadRunOptionsInput, ThreadRunStream, ThreadService,
    },
    AsID, AsRowid, DbEnum, Result,
};
use tracing::error;

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

    async fn get(&self, id: &ID) -> Result<Option<thread::Thread>> {
        let thread = self.db.get_thread(id.as_rowid()?).await?;
        Ok(thread.map(Into::into))
    }

    async fn create_run(
        &self,
        thread_id: &ID,
        options: &ThreadRunOptionsInput,
        yield_thread_created: bool,
    ) -> Result<ThreadRunStream> {
        let Some(answer) = self.answer.clone() else {
            bail!("Answer service is not available");
        };

        let messages: Vec<thread::Message> = self
            .db
            .get_thread_messages(thread_id.as_rowid()?)
            .await?
            .into_iter()
            .flat_map(|x| match x.try_into() {
                Ok(x) => Some(x),
                Err(e) => {
                    error!("Failed to convert thread message: {}", e);
                    None
                }
            })
            .collect();

        let assistant_message_id = self
            .db
            .create_thread_message(
                thread_id.as_rowid()?,
                thread::Role::Assistant.as_enum_str(),
                "",
                None,
                None,
                false,
            )
            .await?;

        let s = answer.answer_v2(&messages, options).await?;

        // Copy ownership of db and thread_id for the stream
        let db = self.db.clone();
        let thread_id = thread_id.clone();
        let s = async_stream::stream! {
            if yield_thread_created {
                yield Ok(ThreadRunItem::thread_created(thread_id.clone()));
            }

            yield Ok(ThreadRunItem::thread_message_created(assistant_message_id.as_id()));

            for await item in s {
                if let Ok(item) = &item {
                    if let Some(code) = &item.thread_message_attachments_code {
                        let code = code
                            .iter()
                            .map(Into::into)
                            .collect::<Vec<_>>();
                        db.update_thread_message_attachments(
                            assistant_message_id,
                            Some(&code),
                            None,
                        ).await?;
                    }

                    if let Some(doc) = &item.thread_message_attachments_doc {
                        let doc = doc
                            .iter()
                            .map(Into::into)
                            .collect::<Vec<_>>();
                        db.update_thread_message_attachments(
                            assistant_message_id,
                            None,
                            Some(&doc),
                        ).await?;
                    }

                    if let Some(content) = &item.thread_message_content_delta {
                        db.append_thread_message_content(
                            assistant_message_id,
                            content,
                        ).await?;
                    }

                    if let Some(relevant_questions) = &item.thread_relevant_questions {
                        db.update_thread_relevant_questions(thread_id.as_rowid()?, relevant_questions).await?;
                    }
                }

                yield item;
            }

            yield Ok(ThreadRunItem::thread_message_completed(assistant_message_id.as_id()));
        };

        Ok(s.boxed())
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

        assert!(service
            .append_messages(
                &thread_id,
                &vec![CreateMessageInput {
                    role: Role::User,
                    content: "This will not success".to_string(),
                    attachments: None,
                }]
            )
            .await
            .is_err());

        assert!(service
            .append_messages(
                &thread_id,
                &vec![CreateMessageInput {
                    role: Role::Assistant,
                    content: "Pong!".to_string(),
                    attachments: None,
                }]
            )
            .await
            .is_ok());
    }
}
