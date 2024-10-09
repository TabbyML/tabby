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
    use tabby_common::{
        api::{
            code::{CodeSearch, CodeSearchParams},
            doc::DocSearch,
        },
        config::AnswerConfig,
    };
    use tabby_db::{testutils::create_user, DbConn};
    use tabby_inference::ChatCompletionStream;
    use tabby_schema::{
        context::ContextService,
        thread::{CreateMessageInput, CreateThreadInput},
    };
    use thread::MessageAttachmentCodeInput;

    use super::*;
    use crate::answer::testutils::{
        FakeChatCompletionStream, FakeCodeSearch, FakeContextService, FakeDocSearch,
    };

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

    #[tokio::test]
    async fn test_get_thread() {
        let db = DbConn::new_in_memory().await.unwrap();
        let user_id = create_user(&db).await.as_id();
        let service = create(db, None);

        let input = CreateThreadInput {
            user_message: CreateMessageInput {
                content: "Ping".to_string(),
                attachments: None,
            },
        };

        let thread_id = service.create(&user_id, &input).await.unwrap();

        let thread = service.get(&thread_id).await.unwrap();
        assert!(thread.is_some());
        assert_eq!(thread.unwrap().id, thread_id);

        let non_existent_id = ID::from("non_existent".to_string());
        let non_existent_thread = service.get(&non_existent_id).await.unwrap();
        assert!(non_existent_thread.is_none());
    }

    #[tokio::test]
    async fn test_set_persisted() {
        let db = DbConn::new_in_memory().await.unwrap();
        let user_id = create_user(&db).await.as_id();
        let service = create(db.clone(), None);

        let input = CreateThreadInput {
            user_message: CreateMessageInput {
                content: "ping".to_string(),
                attachments: None,
            },
        };

        let thread_id = service.create(&user_id, &input).await.unwrap();
        service.set_persisted(&thread_id).await.unwrap();
    }

    pub fn make_code_search_params() -> CodeSearchParams {
        CodeSearchParams {
            min_bm25_score: 0.5,
            min_embedding_score: 0.7,
            min_rrf_score: 0.3,
            num_to_return: 5,
            num_to_score: 10,
        }
    }

    pub fn make_answer_config() -> AnswerConfig {
        AnswerConfig {
            code_search_params: make_code_search_params(),
            presence_penalty: 0.1,
            system_prompt: AnswerConfig::default_system_prompt(),
        }
    }

    #[tokio::test]
    async fn test_create_run() {
        let db = DbConn::new_in_memory().await.unwrap();
        let user_id = create_user(&db).await.as_id();
        let chat: Arc<dyn ChatCompletionStream> = Arc::new(FakeChatCompletionStream);
        let code: Arc<dyn CodeSearch> = Arc::new(FakeCodeSearch);
        let doc: Arc<dyn DocSearch> = Arc::new(FakeDocSearch);
        let context: Arc<dyn ContextService> = Arc::new(FakeContextService);
        let serper = Some(Box::new(FakeDocSearch) as Box<dyn DocSearch>);
        let config = make_answer_config();
        let answer_service = Arc::new(crate::answer::create(
            &config,
            chat.clone(),
            code.clone(),
            doc.clone(),
            context.clone(),
            serper,
        ));
        let service = create(db.clone(), Some(answer_service));

        let input = CreateThreadInput {
            user_message: CreateMessageInput {
                content: "Test message".to_string(),
                attachments: None,
            },
        };

        let thread_id = service.create(&user_id, &input).await.unwrap();

        let policy = AccessPolicy::new(db.clone(), &user_id, false);
        let options = ThreadRunOptionsInput::default();

        let run_stream = service
            .create_run(&policy, &thread_id, &options, None, true, true)
            .await;

        assert!(run_stream.is_ok());
    }

    #[tokio::test]
    async fn test_list_threads() {
        let db = DbConn::new_in_memory().await.unwrap();
        let user_id = create_user(&db).await.as_id();
        let service = create(db, None);

        for i in 0..3 {
            let input = CreateThreadInput {
                user_message: CreateMessageInput {
                    content: format!("Test message {}", i),
                    attachments: None,
                },
            };
            service.create(&user_id, &input).await.unwrap();
        }

        let threads = service
            .list(None, None, None, None, None, None)
            .await
            .unwrap();
        assert_eq!(threads.len(), 3);

        let first_two = service
            .list(None, None, None, None, Some(2), None)
            .await
            .unwrap();
        assert_eq!(first_two.len(), 2);

        let last_two = service
            .list(None, None, None, None, None, Some(2))
            .await
            .unwrap();
        assert_eq!(last_two.len(), 2);
        assert_ne!(first_two[0].id, last_two[0].id);

        let ephemeral_threads = service
            .list(None, Some(true), None, None, None, None)
            .await
            .unwrap();
        assert_eq!(ephemeral_threads.len(), 3);

        service.set_persisted(&threads[0].id).await.unwrap();

        let persisted_threads = service
            .list(None, Some(false), None, None, None, None)
            .await
            .unwrap();
        assert_eq!(persisted_threads.len(), 1);

        let specific_threads = service
            .list(
                Some(&[threads[0].id.clone(), threads[1].id.clone()]),
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(specific_threads.len(), 2);
    }
}
