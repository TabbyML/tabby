use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use juniper::ID;
use tabby_db::{DbConn, ThreadMessageDAO};
use tabby_schema::{
    bail,
    policy::AccessPolicy,
    thread::{
        self, CreateMessageInput, CreateThreadInput, MessageAttachmentInput, ThreadRunItem,
        ThreadRunOptionsInput, ThreadRunStream, ThreadService,
    },
    AsID, AsRowid, DbEnum, Result,
};

use super::{answer::AnswerService, graphql_pagination_to_filter};

struct ThreadServiceImpl {
    db: DbConn,
    answer: Option<Arc<AnswerService>>,
}

impl ThreadServiceImpl {
    async fn get_thread_messages(&self, thread_id: &ID) -> Result<Vec<thread::Message>> {
        let messages = self
            .db
            .list_thread_messages(thread_id.as_rowid()?, None, None, false)
            .await?;
        to_vec_messages(messages)
    }
}

#[async_trait]
impl ThreadService for ThreadServiceImpl {
    async fn create(&self, user_id: &ID, input: &CreateThreadInput) -> Result<ID> {
        let thread_id = self
            .db
            // By default, all new threads are ephemeral
            .create_thread(user_id.as_rowid()?, true)
            .await?;
        self.append_user_message(&thread_id.as_id(), &input.user_message)
            .await?;
        Ok(thread_id.as_id())
    }

    async fn get(&self, id: &ID) -> Result<Option<thread::Thread>> {
        Ok(self
            .list(Some(&[id.clone()]), None, None, None, None, None)
            .await?
            .into_iter()
            .next())
    }

    async fn set_persisted(&self, id: &ID) -> Result<()> {
        self.db
            .update_thread_ephemeral(id.as_rowid()?, false)
            .await?;
        Ok(())
    }

    async fn create_run(
        &self,
        policy: &AccessPolicy,
        thread_id: &ID,
        options: &ThreadRunOptionsInput,
        attachment_input: Option<&MessageAttachmentInput>,
        yield_last_user_message: bool,
        yield_thread_created: bool,
    ) -> Result<ThreadRunStream> {
        let Some(answer) = self.answer.clone() else {
            bail!("Answer service is not available");
        };

        let messages = self.get_thread_messages(thread_id).await?;

        let Some(last_message) = messages.last() else {
            bail!("Thread has no messages");
        };

        if last_message.role != thread::Role::User {
            bail!("Last message in thread is not from user");
        }

        let user_message_id = last_message.id.clone();

        let assistant_message_id = self
            .db
            .create_thread_message(
                thread_id.as_rowid()?,
                thread::Role::Assistant.as_enum_str(),
                "",
                None,
                None,
                None,
                false,
            )
            .await?;

        let s = answer
            .answer_v2(policy, &messages, options, attachment_input)
            .await?;

        // Copy ownership of db and thread_id for the stream
        let db = self.db.clone();
        let thread_id = thread_id.clone();
        let s = async_stream::stream! {
            if yield_thread_created {
                yield Ok(ThreadRunItem::ThreadCreated(thread::ThreadCreated { id: thread_id.clone()}));
            }

            if yield_last_user_message {
                yield Ok(ThreadRunItem::ThreadUserMessageCreated(thread::ThreadUserMessageCreated { id: user_message_id }));
            }

            yield Ok(ThreadRunItem::ThreadAssistantMessageCreated(thread::ThreadAssistantMessageCreated { id: assistant_message_id.as_id() }));

            for await item in s {
                match &item {
                    Ok(ThreadRunItem::ThreadAssistantMessageContentDelta(x)) => {
                        db.append_thread_message_content(assistant_message_id, &x.delta).await?;
                    }

                    Ok(ThreadRunItem::ThreadAssistantMessageAttachmentsCode(x)) => {
                        let code = x
                            .hits
                            .iter()
                            .map(|x| (&x.code).into())
                            .collect::<Vec<_>>();
                        db.update_thread_message_code_attachments(
                            assistant_message_id,
                            &code,
                        ).await?;
                    }

                    Ok(ThreadRunItem::ThreadAssistantMessageAttachmentsDoc(x)) => {
                        let doc = x
                            .hits
                            .iter()
                            .map(|x| (&x.doc).into())
                            .collect::<Vec<_>>();
                        db.update_thread_message_doc_attachments(
                            assistant_message_id,
                            &doc,
                        ).await?;
                    }

                    Ok(ThreadRunItem::ThreadRelevantQuestions(x)) => {
                        db.update_thread_relevant_questions(thread_id.as_rowid()?, &x.questions).await?;
                    }

                    _ => {}
                }

                yield item;
            }

            yield Ok(ThreadRunItem::ThreadAssistantMessageCompleted(thread::ThreadAssistantMessageCompleted { id: assistant_message_id.as_id() }));
        };

        Ok(s.boxed())
    }

    async fn append_user_message(
        &self,
        thread_id: &ID,
        message: &CreateMessageInput,
    ) -> Result<()> {
        let thread_id = thread_id.as_rowid()?;
        let client_code = message.attachments.as_ref().and_then(|x| {
            let code = x.code.iter().map(Into::into).collect::<Vec<_>>();
            // If there are no code attachments, return None
            if code.is_empty() {
                None
            } else {
                Some(code)
            }
        });

        self.db
            .create_thread_message(
                thread_id,
                thread::Role::User.as_enum_str(),
                &message.content,
                None,
                client_code.as_deref(),
                None,
                true,
            )
            .await?;

        Ok(())
    }

    async fn list(
        &self,
        ids: Option<&[ID]>,
        is_ephemeral: Option<bool>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<thread::Thread>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;

        let ids = ids.map(|x| {
            x.iter()
                .filter_map(|x| x.as_rowid().ok())
                .collect::<Vec<_>>()
        });
        let threads = self
            .db
            .list_threads(ids.as_deref(), is_ephemeral, limit, skip_id, backwards)
            .await?;

        Ok(threads.into_iter().map(Into::into).collect())
    }

    async fn list_thread_messages(
        &self,
        thread_id: &ID,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<thread::Message>> {
        let thread_id = thread_id.as_rowid()?;
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;

        let messages = self
            .db
            .list_thread_messages(thread_id, limit, skip_id, backwards)
            .await?;

        to_vec_messages(messages)
    }

    async fn delete_thread_message_pair(
        &self,
        thread_id: &ID,
        user_message_id: &ID,
        assistant_message_id: &ID,
    ) -> Result<()> {
        self.db
            .delete_thread_message_pair(
                thread_id.as_rowid()?,
                user_message_id.as_rowid()?,
                assistant_message_id.as_rowid()?,
            )
            .await?;
        Ok(())
    }
}

fn to_vec_messages(messages: Vec<ThreadMessageDAO>) -> Result<Vec<thread::Message>> {
    let mut output = vec![];
    output.reserve(messages.len());

    for x in messages {
        let message: thread::Message = x.try_into()?;
        output.push(message);
    }

    Ok(output)
}

pub fn create(db: DbConn, answer: Option<Arc<AnswerService>>) -> impl ThreadService {
    ThreadServiceImpl { db, answer }
}

#[cfg(test)]
mod tests {
    use tabby_db::{testutils::create_user, DbConn};
    use tabby_schema::thread::{CreateMessageInput, CreateThreadInput};
    use thread::MessageAttachmentCodeInput;

    use super::*;

    #[tokio::test]
    async fn test_create_thread() {
        let db = DbConn::new_in_memory().await.unwrap();
        let user_id = create_user(&db).await.as_id();
        let service = create(db, None);

        let input = CreateThreadInput {
            user_message: CreateMessageInput {
                content: "Hello, world!".to_string(),
                attachments: None,
            },
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
                    user_message: CreateMessageInput {
                        content: "Ping!".to_string(),
                        attachments: Some(MessageAttachmentInput {
                            code: vec![MessageAttachmentCodeInput {
                                filepath: Some("main.rs".to_string()),
                                content: "fn main() { println!(\"Hello, world!\"); }".to_string(),
                                start_line: Some(1),
                            }],
                        }),
                    },
                },
            )
            .await
            .unwrap();

        let messages = service
            .list_thread_messages(&thread_id, None, None, None, None)
            .await
            .unwrap();
        assert_eq!(
            messages[0].attachment.client_code[0].filepath,
            Some("main.rs".to_string())
        );

        assert!(service
            .append_user_message(
                &thread_id,
                &CreateMessageInput {
                    content: "Pong!".to_string(),
                    attachments: None,
                }
            )
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_delete_thread_message_pair() {
        let db = DbConn::new_in_memory().await.unwrap();
        let user_id = create_user(&db).await.as_id();
        let service = create(db.clone(), None);

        let thread_id = service
            .create(
                &user_id,
                &CreateThreadInput {
                    user_message: CreateMessageInput {
                        content: "Ping!".to_string(),
                        attachments: None,
                    },
                },
            )
            .await
            .unwrap();

        let assistant_message_id = db
            .create_thread_message(
                thread_id.as_rowid().unwrap(),
                thread::Role::Assistant.as_enum_str(),
                "Pong!",
                None,
                None,
                None,
                false,
            )
            .await
            .unwrap();

        let user_message_id = assistant_message_id - 1;

        // Create another user message to test the error case
        let another_user_message_id = db
            .create_thread_message(
                thread_id.as_rowid().unwrap(),
                thread::Role::User.as_enum_str(),
                "Ping another time!",
                None,
                None,
                None,
                false,
            )
            .await
            .unwrap();

        let messages = service
            .list_thread_messages(&thread_id, None, None, None, None)
            .await
            .unwrap();
        assert_eq!(messages.len(), 3);

        assert!(service
            .delete_thread_message_pair(
                &thread_id,
                &another_user_message_id.as_id(),
                &assistant_message_id.as_id()
            )
            .await
            .is_err());

        assert!(service
            .delete_thread_message_pair(
                &thread_id,
                &assistant_message_id.as_id(),
                &another_user_message_id.as_id()
            )
            .await
            .is_err());

        assert!(service
            .delete_thread_message_pair(
                &thread_id,
                &user_message_id.as_id(),
                &assistant_message_id.as_id()
            )
            .await
            .is_ok());

        // Verify that the messages were deleted
        let messages = service
            .list_thread_messages(&thread_id, None, None, None, None)
            .await
            .unwrap();
        assert_eq!(messages.len(), 1);
    }
}
