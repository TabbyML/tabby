mod prompt_tools;

use std::sync::Arc;

use anyhow::anyhow;
use async_openai_alt::{error::OpenAIError, types::CreateChatCompletionRequestArgs};
use async_stream::stream;
use futures::stream::BoxStream;
use prompt_tools::{pipeline_decide_need_codebase_context, pipeline_related_questions};
use tabby_common::{api::structured_doc::DocSearchDocument, config::AnswerConfig};
use tabby_inference::ChatCompletionStream;
use tabby_schema::{
    auth::AuthenticationService,
    context::{ContextInfoHelper, ContextService},
    policy::AccessPolicy,
    repository::{Repository, RepositoryService},
    thread::{
        CodeQueryInput, MessageAttachment, MessageAttachmentCodeFileList, MessageAttachmentDoc,
        MessageDocSearchHit, ThreadAssistantMessageAttachmentsCode,
        ThreadAssistantMessageAttachmentsCodeFileList, ThreadAssistantMessageAttachmentsDoc,
        ThreadAssistantMessageContentDelta, ThreadRelevantQuestions, ThreadRunItem,
        ThreadRunOptionsInput,
    },
};
use tracing::{debug, error, warn};

use crate::service::{
    retrieval::RetrievalService,
    utils::{
        convert_messages_to_chat_completion_request,
        convert_user_message_to_chat_completion_request,
    },
};
pub struct AnswerService {
    config: AnswerConfig,
    auth: Arc<dyn AuthenticationService>,
    chat: Arc<dyn ChatCompletionStream>,
    retrieval: Arc<RetrievalService>,
    context: Arc<dyn ContextService>,
    repository: Arc<dyn RepositoryService>,
}

impl AnswerService {
    fn new(
        config: &AnswerConfig,
        auth: Arc<dyn AuthenticationService>,
        chat: Arc<dyn ChatCompletionStream>,
        retrieval: Arc<RetrievalService>,
        context: Arc<dyn ContextService>,
        repository: Arc<dyn RepositoryService>,
    ) -> Self {
        Self {
            config: config.clone(),
            auth,
            chat,
            retrieval,
            context,
            repository,
        }
    }

    pub async fn answer<'a>(
        self: Arc<Self>,
        policy: &AccessPolicy,
        messages: &[tabby_schema::thread::Message],
        options: &ThreadRunOptionsInput,
        user_attachment_input: Option<&tabby_schema::thread::MessageAttachmentInput>,
    ) -> tabby_schema::Result<BoxStream<'a, tabby_schema::Result<ThreadRunItem>>> {
        let (last_message, messages) = match messages.split_last() {
            Some((last_message, messages)) => (last_message.clone(), messages.to_vec()),
            None => {
                return Err(anyhow!("No message found in the request").into());
            }
        };

        let options = options.clone();
        let user_attachment_input = user_attachment_input.cloned();
        let policy = policy.clone();

        let s = stream! {
            let context_info = self.context.read(Some(&policy)).await?;
            let context_info_helper = context_info.helper();

            let mut attachment = MessageAttachment::default();

            // 1. Collect relevant code if needed.
            if let Some(code_query) = options.code_query.as_ref() {
                if let Some(repository) = self.find_repository(&context_info_helper, code_query, policy.clone()).await {
                    let need_codebase_context = pipeline_decide_need_codebase_context(self.chat.clone(), &last_message.content).await?;
                    yield Ok(ThreadRunItem::ThreadAssistantMessageReadingCode(need_codebase_context.clone()));
                    if need_codebase_context.file_list {
                        // List at most 300 files in the repository.
                        match self.retrieval.collect_file_list(&policy, &repository, None, Some(300)).await {
                            Ok((file_list, truncated)) => {
                                attachment.code_file_list = Some(MessageAttachmentCodeFileList {
                                    file_list: file_list.clone(),
                                    truncated,
                                });
                                yield Ok(ThreadRunItem::ThreadAssistantMessageAttachmentsCodeFileList(ThreadAssistantMessageAttachmentsCodeFileList {
                                    file_list,
                                    truncated
                                }));
                            }
                            Err(e) => {
                                error!("failed to list files for repository {}: {}", repository.id, e);
                            }
                        }
                    }

                    if need_codebase_context.snippet {
                        let hits = self.retrieval.collect_relevant_code(
                            &repository,
                            &context_info_helper,
                            code_query,
                            &self.config.code_search_params,
                            options.debug_options.as_ref().and_then(|x| x.code_search_params_override.as_ref()),
                        ).await;
                        attachment.code = hits.iter().map(|x| x.doc.clone().into()).collect::<Vec<_>>();

                        if !hits.is_empty() {
                            let hits = hits.into_iter().map(|x| x.into()).collect::<Vec<_>>();
                            yield Ok(ThreadRunItem::ThreadAssistantMessageAttachmentsCode(
                                ThreadAssistantMessageAttachmentsCode { hits }
                            ));
                        }
                    }

                };
            };

            // 2. Collect relevant docs if needed.
            if let Some(doc_query) = options.doc_query.as_ref() {
                let hits = self.retrieval.collect_relevant_docs(&context_info_helper, doc_query)
                    .await;
                attachment.doc = futures::future::join_all(hits.iter().map(|x| async {
                    Self::new_message_attachment_doc(self.auth.clone(), x.doc.clone()).await
                })).await;

                debug!("query content: {:?}, matched {:?} docs", doc_query.content, attachment.doc.len());

                if !attachment.doc.is_empty() {
                    let hits = futures::future::join_all(hits.into_iter().map(|x| {
                        let score = x.score;
                        let doc = x.doc.clone();
                        let auth = self.auth.clone();
                        async move {
                            MessageDocSearchHit {
                                score: score as f64,
                                doc: Self::new_message_attachment_doc(auth, doc).await,
                            }
                        }
                    })).await;
                    yield Ok(ThreadRunItem::ThreadAssistantMessageAttachmentsDoc(
                        ThreadAssistantMessageAttachmentsDoc { hits }
                    ));
                }
            };

            // 3. Generate relevant questions.
            if options.generate_relevant_questions {
                // Rewrite [[source:${id}]] tags to the actual source name for generate relevant questions.
                let content = context_info_helper.rewrite_tag(&last_message.content);
                match self
                    .generate_relevant_questions(&attachment, &content)
                    .await{
                    Ok(questions) => {
                        yield Ok(ThreadRunItem::ThreadRelevantQuestions(ThreadRelevantQuestions{
                            questions
                        }));
                    }
                    Err(err) => {
                        warn!("Failed to generate relevant questions: {}", err);
                    }
                }
            }

            // 4. Prepare requesting LLM
            let request = {
                let mut chat_messages = convert_messages_to_chat_completion_request(
                    Some(&self.config.system_prompt),
                    &context_info_helper,
                    &messages,
                )?;
                let user_message = convert_user_message_to_chat_completion_request(
                    &context_info_helper,
                    &last_message.content,
                    &attachment,
                    user_attachment_input.as_ref(),
                );
                chat_messages.push(user_message);

                CreateChatCompletionRequestArgs::default()
                    .messages(chat_messages)
                    .model(options.model_name.as_deref().unwrap_or(""))
                    .presence_penalty(self.config.presence_penalty)
                    .build()
                    .expect("Failed to build chat completion request")
            };

            let s = match self.chat.chat_stream(request).await {
                Ok(s) => s,
                Err(err) => {
                    warn!("Failed to create chat completion stream: {:?}", err);
                    return;
                }
            };

            for await chunk in s {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(err) => {
                        if let OpenAIError::StreamError(content) = &err {
                            if content == "Stream ended" {
                                break;
                            }
                        }
                        error!("Failed to get chat completion chunk: {:?}", err);
                        yield Err(anyhow!("Failed to get chat completion chunk: {:?}", err).into());
                        return;
                    }
                };

                let content = chunk.choices.first().and_then(|x| x.delta.content.as_deref());
                if let Some(content) = content {
                    yield Ok(ThreadRunItem::ThreadAssistantMessageContentDelta(ThreadAssistantMessageContentDelta {
                        delta: content.to_owned()
                    }));
                }
            }
        };

        Ok(Box::pin(s))
    }

    async fn new_message_attachment_doc(
        auth: Arc<dyn AuthenticationService>,
        doc: DocSearchDocument,
    ) -> MessageAttachmentDoc {
        let (author, committer) = match &doc {
            DocSearchDocument::Issue(issue) => (issue.author_email.as_deref(), None),
            DocSearchDocument::Pull(pull) => (pull.author_email.as_deref(), None),
            DocSearchDocument::Commit(commit) => (
                Some(commit.author_email.as_str()),
                Some(commit.committer_email.as_str()),
            ),
            _ => (None, None),
        };
        let user = if let Some(email) = author {
            auth.get_user_by_email(email).await.ok().map(|x| x.into())
        } else {
            None
        };

        let committer = if let Some(email) = committer {
            auth.get_user_by_email(email).await.ok().map(|x| x.into())
        } else {
            None
        };

        MessageAttachmentDoc::from_doc_search_document(doc, user, committer)
    }

    async fn find_repository(
        &self,
        helper: &ContextInfoHelper,
        input: &CodeQueryInput,
        policy: AccessPolicy,
    ) -> Option<Repository> {
        let source_id = {
            if let Some(source_id) = &input.source_id {
                if helper.can_access_source_id(source_id) {
                    Some(source_id.as_str())
                } else {
                    None
                }
            } else if let Some(git_url) = &input.git_url {
                helper.allowed_code_repository().closest_match(git_url)
            } else {
                None
            }
        }?;

        match self.repository.repository_list(Some(&policy)).await {
            Ok(repos) => repos.into_iter().find(|x| x.source_id == source_id),
            Err(_) => None,
        }
    }

    async fn generate_relevant_questions(
        &self,
        attachment: &MessageAttachment,
        question: &str,
    ) -> anyhow::Result<Vec<String>> {
        if attachment.code.is_empty() && attachment.doc.is_empty() {
            return Ok(vec![]);
        }

        let snippets: Vec<String> = attachment
            .code
            .iter()
            .map(|snippet| {
                format!(
                    "```{} title=\"{}\"\n{}\n```",
                    snippet.language, snippet.filepath, snippet.content
                )
            })
            .chain(
                attachment
                    .doc
                    .iter()
                    .map(|doc| format!("```\n{}\n```", doc.content())),
            )
            .collect();

        let context: String = snippets.join("\n\n");
        pipeline_related_questions(self.chat.clone(), &context, question).await
    }
}

pub fn create(
    config: &AnswerConfig,
    auth: Arc<dyn AuthenticationService>,
    chat: Arc<dyn ChatCompletionStream>,
    retrieval: Arc<RetrievalService>,
    context: Arc<dyn ContextService>,
    repository: Arc<dyn RepositoryService>,
) -> AnswerService {
    AnswerService::new(config, auth, chat, retrieval, context, repository)
}

#[cfg(test)]
pub mod testutils;

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use juniper::ID;
    use tabby_common::{
        api::{
            code::{CodeSearch, CodeSearchParams},
            structured_doc::DocSearch,
        },
        config::AnswerConfig,
    };
    use tabby_db::DbConn;
    use tabby_inference::ChatCompletionStream;
    use tabby_schema::{
        context::{ContextInfo, ContextInfoHelper, ContextService, ContextSourceValue},
        repository::{Repository, RepositoryKind},
        thread::{CodeQueryInput, MessageAttachment},
        web_documents::PresetWebDocument,
        AsID,
    };

    use super::{
        testutils::{
            make_repository_service, FakeChatCompletionStream, FakeCodeSearch, FakeContextService,
            FakeDocSearch,
        },
        *,
    };
    use crate::{
        retrieval,
        service::{access_policy::testutils::make_policy, auth},
        utils::build_user_prompt,
    };

    const TEST_SOURCE_ID: &str = "source-1";
    const TEST_GIT_URL: &str = "TabbyML/tabby";
    const TEST_FILEPATH: &str = "test.rs";
    const TEST_LANGUAGE: &str = "rust";
    const TEST_CONTENT: &str = "fn main() {}";

    pub fn make_answer_config() -> AnswerConfig {
        AnswerConfig {
            code_search_params: make_code_search_params(),
            presence_penalty: 0.1,
            system_prompt: AnswerConfig::default_system_prompt(),
        }
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
    pub fn make_code_query_input(source_id: Option<&str>, git_url: Option<&str>) -> CodeQueryInput {
        CodeQueryInput {
            filepath: Some(TEST_FILEPATH.to_string()),
            content: TEST_CONTENT.to_string(),
            git_url: git_url.map(|url| url.to_string()),
            source_id: source_id.map(|id| id.to_string()),
            language: Some(TEST_LANGUAGE.to_string()),
        }
    }

    pub fn make_message(
        id: i32,
        content: &str,
        role: tabby_schema::thread::Role,
        attachment: Option<tabby_schema::thread::MessageAttachment>,
    ) -> tabby_schema::thread::Message {
        tabby_schema::thread::Message {
            id: id.as_id(),
            thread_id: ID::new("0"),
            code_source_id: None,
            content: content.to_owned(),
            role,
            attachment: attachment.unwrap_or_default(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_build_user_prompt() {
        let user_input = "What is the purpose of this code?";
        let assistant_attachment = tabby_schema::thread::MessageAttachment {
            doc: vec![tabby_schema::thread::MessageAttachmentDoc::Web(tabby_schema::thread::MessageAttachmentWebDoc {
                title: "Documentation".to_owned(),
                content: "This code implements a basic web server.".to_owned(),
                link: "https://example.com/docs".to_owned(),
            })],
            code: vec![tabby_schema::thread::MessageAttachmentCode {
                git_url: "https://github.com/".to_owned(),
                commit: Some("commit".to_owned()),
                filepath: "server.py".to_owned(),
                language: "python".to_owned(),
                content: "from flask import Flask\n\napp = Flask(__name__)\n\n@app.route('/')\ndef hello():\n    return 'Hello, World!'".to_owned(),
                start_line: Some(1),
            }],
            client_code: vec![],
            code_file_list: None,
        };
        let user_attachment_input = None;

        let prompt = build_user_prompt(user_input, &assistant_attachment, user_attachment_input);

        println!("{}", prompt.as_str());
        assert!(prompt.contains(user_input));
        assert!(prompt.contains("This code implements a basic web server."));
        assert!(prompt.contains("from flask import Flask"));
        assert!(prompt.contains("[[citation:1]]"));
        assert!(prompt.contains("[[citation:2]]"));
    }

    #[test]
    fn test_convert_messages_to_chat_completion_request() {
        // Fake assistant attachment
        let attachment = tabby_schema::thread::MessageAttachment {
            doc: vec![tabby_schema::thread::MessageAttachmentDoc::Web(
                tabby_schema::thread::MessageAttachmentWebDoc {
                    title: "1. Example Document".to_owned(),
                    content: "This is an example".to_owned(),
                    link: "https://example.com".to_owned(),
                },
            )],
            code: vec![tabby_schema::thread::MessageAttachmentCode {
                git_url: "https://github.com".to_owned(),
                commit: Some("commit".to_owned()),
                filepath: "server.py".to_owned(),
                language: "python".to_owned(),
                content: "print('Hello, server!')".to_owned(),
                start_line: Some(1),
            }],
            client_code: vec![tabby_schema::thread::MessageAttachmentClientCode {
                filepath: Some("client.py".to_owned()),
                content: "print('Hello, client!')".to_owned(),
                start_line: Some(1),
            }],
            code_file_list: Some(MessageAttachmentCodeFileList {
                file_list: vec!["client.py".to_owned(), "server.py".to_owned()],
                truncated: false,
            }),
        };

        let messages = vec![
            make_message(1, "Hello", tabby_schema::thread::Role::User, None),
            make_message(
                2,
                "Hi, [[source:preset_web_document:source-1]], [[source:2]]",
                tabby_schema::thread::Role::Assistant,
                Some(attachment),
            ),
        ];
        let last_message = make_message(3, "How are you?", tabby_schema::thread::Role::User, None);

        let user_attachment_input = tabby_schema::thread::MessageAttachmentInput {
            code: vec![tabby_schema::thread::MessageAttachmentCodeInput {
                filepath: Some("client.py".to_owned()),
                content: "print('Hello, client!')".to_owned(),
                start_line: Some(1),
            }],
        };

        let context_info = ContextInfo {
            sources: vec![ContextSourceValue::PresetWebDocument(PresetWebDocument {
                id: ID::from("id".to_owned()),
                name: "source-1".into(),
                updated_at: None,
                job_info: None,
                is_active: true,
            })],
        };

        let rewriter = context_info.helper();

        let config = make_answer_config();
        let mut output = super::convert_messages_to_chat_completion_request(
            Some(&config.system_prompt),
            &rewriter,
            &messages,
        )
        .unwrap();
        let user_message = convert_user_message_to_chat_completion_request(
            &rewriter,
            &last_message.content,
            &tabby_schema::thread::MessageAttachment::default(),
            Some(&user_attachment_input),
        );
        output.push(user_message);

        insta::assert_yaml_snapshot!(output);
    }

    #[tokio::test]
    async fn test_generate_relevant_questions() {
        let auth = Arc::new(auth::testutils::FakeAuthService::new(vec![]));
        let chat: Arc<dyn ChatCompletionStream> = Arc::new(FakeChatCompletionStream {
            return_error: false,
        });
        let code: Arc<dyn CodeSearch> = Arc::new(FakeCodeSearch);
        let doc: Arc<dyn DocSearch> = Arc::new(FakeDocSearch);
        let context: Arc<dyn ContextService> = Arc::new(FakeContextService);
        let serper = Some(Box::new(FakeDocSearch) as Box<dyn DocSearch>);
        let config = make_answer_config();
        let db = DbConn::new_in_memory().await.unwrap();
        let repo = make_repository_service(db).await.unwrap();

        let retrieval = Arc::new(retrieval::create(
            code.clone(),
            doc.clone(),
            serper,
            repo.clone(),
        ));
        let service = AnswerService::new(&config, auth, chat, retrieval, context, repo);

        let attachment = MessageAttachment {
            doc: vec![tabby_schema::thread::MessageAttachmentDoc::Web(
                tabby_schema::thread::MessageAttachmentWebDoc {
                    title: "1. Example Document".to_owned(),
                    content: "This is an example".to_owned(),
                    link: "https://example.com".to_owned(),
                },
            )],
            code: vec![tabby_schema::thread::MessageAttachmentCode {
                git_url: "https://github.com".to_owned(),
                commit: Some("commit".to_owned()),
                filepath: "server.py".to_owned(),
                language: "python".to_owned(),
                content: "print('Hello, server!')".to_owned(),
                start_line: Some(1),
            }],
            client_code: vec![tabby_schema::thread::MessageAttachmentClientCode {
                filepath: Some("client.py".to_owned()),
                content: "print('Hello, client!')".to_owned(),
                start_line: Some(1),
            }],
            code_file_list: None,
        };

        let question = "What is the purpose of this code?";

        let result = service
            .generate_relevant_questions(&attachment, question)
            .await;

        let expected = vec![
            "What is the main functionality of the provided code?".to_string(),
            "How does the code snippet implement a web server?".to_string(),
            "Can you explain how the Flask app works in this context?".to_string(),
        ];

        assert_eq!(result.unwrap(), expected);
    }

    #[tokio::test]
    async fn test_generate_relevant_questions_error() {
        let auth = Arc::new(auth::testutils::FakeAuthService::new(vec![]));
        let chat: Arc<dyn ChatCompletionStream> =
            Arc::new(FakeChatCompletionStream { return_error: true });
        let code: Arc<dyn CodeSearch> = Arc::new(FakeCodeSearch);
        let doc: Arc<dyn DocSearch> = Arc::new(FakeDocSearch);
        let context: Arc<dyn ContextService> = Arc::new(FakeContextService);
        let serper = Some(Box::new(FakeDocSearch) as Box<dyn DocSearch>);
        let config = make_answer_config();
        let db = DbConn::new_in_memory().await.unwrap();
        let repo = make_repository_service(db).await.unwrap();

        let retrieval = Arc::new(retrieval::create(
            code.clone(),
            doc.clone(),
            serper,
            repo.clone(),
        ));
        let service = AnswerService::new(&config, auth, chat, retrieval, context, repo);

        let attachment = MessageAttachment {
            doc: vec![tabby_schema::thread::MessageAttachmentDoc::Web(
                tabby_schema::thread::MessageAttachmentWebDoc {
                    title: "1. Example Document".to_owned(),
                    content: "This is an example".to_owned(),
                    link: "https://example.com".to_owned(),
                },
            )],
            code: vec![tabby_schema::thread::MessageAttachmentCode {
                git_url: "https://github.com".to_owned(),
                commit: Some("commit".to_owned()),
                filepath: "server.py".to_owned(),
                language: "python".to_owned(),
                content: "print('Hello, server!')".to_owned(),
                start_line: Some(1),
            }],
            client_code: vec![tabby_schema::thread::MessageAttachmentClientCode {
                filepath: Some("client.py".to_owned()),
                content: "print('Hello, client!')".to_owned(),
                start_line: Some(1),
            }],
            code_file_list: None,
        };

        let question = "What is the purpose of this code?";

        let result = service
            .generate_relevant_questions(&attachment, question)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_answer() {
        use std::sync::Arc;

        use futures::StreamExt;
        use tabby_schema::{policy::AccessPolicy, thread::ThreadRunOptionsInput};

        let auth = Arc::new(auth::testutils::FakeAuthService::new(vec![]));
        let chat: Arc<dyn ChatCompletionStream> = Arc::new(FakeChatCompletionStream {
            return_error: false,
        });
        let code: Arc<dyn CodeSearch> = Arc::new(FakeCodeSearch);
        let doc: Arc<dyn DocSearch> = Arc::new(FakeDocSearch);
        let context: Arc<dyn ContextService> = Arc::new(FakeContextService);
        let serper = Some(Box::new(FakeDocSearch) as Box<dyn DocSearch>);

        let config = AnswerConfig {
            code_search_params: make_code_search_params(),
            presence_penalty: 0.1,
            system_prompt: AnswerConfig::default_system_prompt(),
        };
        let db = DbConn::new_in_memory().await.unwrap();
        let repo = make_repository_service(db).await.unwrap();
        let retrieval = Arc::new(retrieval::create(
            code.clone(),
            doc.clone(),
            serper,
            repo.clone(),
        ));
        let service = Arc::new(AnswerService::new(
            &config, auth, chat, retrieval, context, repo,
        ));

        let db = DbConn::new_in_memory().await.unwrap();
        let policy = AccessPolicy::new(db, &1.as_id(), false);
        let messages = vec![
            make_message(1, "What is Rust?", tabby_schema::thread::Role::User, None),
            make_message(
                2,
                "Rust is a systems programming language.",
                tabby_schema::thread::Role::Assistant,
                None,
            ),
            make_message(
                3,
                "Can you explain more about Rust's memory safety?",
                tabby_schema::thread::Role::User,
                None,
            ),
        ];
        let options = ThreadRunOptionsInput {
            model_name: None,
            code_query: Some(make_code_query_input(
                Some(TEST_SOURCE_ID),
                Some(TEST_GIT_URL),
            )),
            doc_query: Some(tabby_schema::thread::DocQueryInput {
                content: "Rust memory safety".to_string(),
                source_ids: Some(vec![TEST_SOURCE_ID.to_string()]),
                search_public: true,
            }),
            generate_relevant_questions: true,
            debug_options: None,
        };
        let user_attachment_input = None;

        let result = service
            .answer(&policy, &messages, &options, user_attachment_input)
            .await
            .unwrap();

        let collected_results: Vec<_> = result.collect().await;

        assert_eq!(
            collected_results.len(),
            4,
            "Expected 4 items in the result stream"
        );
    }

    #[tokio::test]
    async fn test_find_repository() {
        // Setup test database
        let db = DbConn::new_in_memory().await.unwrap();
        let repo_service = make_repository_service(db.clone()).await.unwrap();

        // Create test repository via git service
        let repo_id = repo_service
            .git()
            .create("test-repo".to_string(), TEST_GIT_URL.to_string())
            .await
            .unwrap();
        let source_id = format!("git:{}", repo_id);

        println!("Created test repository with ID: {}", source_id);

        // Setup test context with actual repository
        let test_repo = Repository {
            id: repo_id,
            source_id: source_id.clone(),
            name: "test-repo".to_string(),
            kind: RepositoryKind::Git,
            dir: PathBuf::from("test-repo"),
            git_url: TEST_GIT_URL.to_string(),
            refs: vec![],
        };

        let context_info = ContextInfo {
            sources: vec![ContextSourceValue::Repository(test_repo)],
        };
        let context_info_helper = ContextInfoHelper::new(&context_info);

        let policy = make_policy(db.clone()).await;

        let repos = repo_service.repository_list(Some(&policy)).await.unwrap();
        assert!(!repos.is_empty(), "Repository should exist");

        let auth = Arc::new(auth::testutils::FakeAuthService::new(vec![]));
        let chat = Arc::new(FakeChatCompletionStream {
            return_error: false,
        });
        let code = Arc::new(FakeCodeSearch);
        let doc = Arc::new(FakeDocSearch);
        let context = Arc::new(FakeContextService);
        let serper = Some(Box::new(FakeDocSearch) as Box<dyn DocSearch>);
        let config = make_answer_config();

        let retrieval = Arc::new(retrieval::create(
            code.clone(),
            doc.clone(),
            serper,
            repo_service.clone(),
        ));
        let service = AnswerService::new(&config, auth, chat, retrieval, context, repo_service);

        // Test repository lookup
        let input = make_code_query_input(Some(&source_id), Some(TEST_GIT_URL));

        let result = service
            .find_repository(&context_info_helper, &input, policy)
            .await;

        assert!(result.is_some(), "Should find repository");
        let found_repo = result.unwrap();
        assert_eq!(found_repo.source_id, source_id, "Source ID should match");
        assert_eq!(found_repo.git_url, TEST_GIT_URL, "Git URL should match");
        assert_eq!(found_repo.kind, RepositoryKind::Git, "Kind should be Git");
    }
}
