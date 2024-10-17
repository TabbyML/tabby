use std::sync::Arc;

use anyhow::anyhow;
use async_openai::{
    error::OpenAIError,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs, Role,
    },
};
use async_stream::stream;
use futures::stream::BoxStream;
use tabby_common::{
    api::{
        code::{CodeSearch, CodeSearchError, CodeSearchHit, CodeSearchParams, CodeSearchQuery},
        doc::{DocSearch, DocSearchError, DocSearchHit},
    },
    config::AnswerConfig,
};
use tabby_inference::ChatCompletionStream;
use tabby_schema::{
    context::{ContextInfoHelper, ContextService},
    policy::AccessPolicy,
    thread::{
        self, CodeQueryInput, CodeSearchParamsOverrideInput, DocQueryInput, MessageAttachment,
        ThreadAssistantMessageAttachmentsCode, ThreadAssistantMessageAttachmentsDoc,
        ThreadAssistantMessageContentDelta, ThreadRelevantQuestions, ThreadRunItem,
        ThreadRunOptionsInput,
    },
};
use tracing::{debug, error, warn};

use crate::bail;

pub struct AnswerService {
    config: AnswerConfig,
    chat: Arc<dyn ChatCompletionStream>,
    code: Arc<dyn CodeSearch>,
    doc: Arc<dyn DocSearch>,
    context: Arc<dyn ContextService>,
    serper: Option<Box<dyn DocSearch>>,
}

impl AnswerService {
    fn new(
        config: &AnswerConfig,
        chat: Arc<dyn ChatCompletionStream>,
        code: Arc<dyn CodeSearch>,
        doc: Arc<dyn DocSearch>,
        context: Arc<dyn ContextService>,
        serper: Option<Box<dyn DocSearch>>,
    ) -> Self {
        Self {
            config: config.clone(),
            chat,
            code,
            doc,
            context,
            serper,
        }
    }

    pub async fn answer_v2<'a>(
        self: Arc<Self>,
        policy: &AccessPolicy,
        messages: &[tabby_schema::thread::Message],
        options: &ThreadRunOptionsInput,
        user_attachment_input: Option<&tabby_schema::thread::MessageAttachmentInput>,
    ) -> tabby_schema::Result<BoxStream<'a, tabby_schema::Result<ThreadRunItem>>> {
        let messages = messages.to_vec();
        let options = options.clone();
        let user_attachment_input = user_attachment_input.cloned();
        let policy = policy.clone();

        let s = stream! {
            let context_info = self.context.read(Some(&policy)).await?;
            let context_info_helper = context_info.helper();

            let query = match messages.last() {
                Some(query) => query,
                None => {
                    yield Err(anyhow!("No query found in the request").into());
                    return;
                }
            };

            let mut attachment = MessageAttachment::default();

            // 1. Collect relevant code if needed.
            if let Some(code_query) = options.code_query.as_ref() {
                let hits = self.collect_relevant_code(
                    &context_info_helper,
                    code_query,
                    &self.config.code_search_params,
                    options.debug_options.as_ref().and_then(|x| x.code_search_params_override.as_ref())
                ).await;
                attachment.code = hits.iter().map(|x| x.doc.clone().into()).collect::<Vec<_>>();

                if !hits.is_empty() {
                    let hits = hits.into_iter().map(|x| x.into()).collect::<Vec<_>>();
                    yield Ok(ThreadRunItem::ThreadAssistantMessageAttachmentsCode(
                        ThreadAssistantMessageAttachmentsCode { hits }
                    ));
                }
            };

            // 2. Collect relevant docs if needed.
            if let Some(doc_query) = options.doc_query.as_ref() {
                let hits = self.collect_relevant_docs(&context_info_helper, doc_query)
                    .await;
                attachment.doc = hits.iter()
                        .map(|x| x.doc.clone().into())
                        .collect::<Vec<_>>();

                if !attachment.doc.is_empty() {
                    let hits = hits.into_iter().map(|x| x.into()).collect::<Vec<_>>();
                    yield Ok(ThreadRunItem::ThreadAssistantMessageAttachmentsDoc(
                        ThreadAssistantMessageAttachmentsDoc { hits }
                    ));
                }
            };

            // 3. Generate relevant questions.
            if options.generate_relevant_questions {
                // Rewrite [[source:${id}]] tags to the actual source name for generate relevant questions.
                let content = context_info_helper.rewrite_tag(&query.content);
                let questions = self
                    .generate_relevant_questions_v2(&attachment, &content)
                    .await;
                yield Ok(ThreadRunItem::ThreadRelevantQuestions(ThreadRelevantQuestions{
                    questions
                }));
            }

            // 4. Prepare requesting LLM
            let request = {
                let chat_messages = convert_messages_to_chat_completion_request(&self.config, &context_info_helper, &messages, &attachment, user_attachment_input.as_ref())?;

                CreateChatCompletionRequestArgs::default()
                    .messages(chat_messages)
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
                        if let OpenAIError::StreamError(content) = err {
                            if content == "Stream ended" {
                                break;
                            }
                        } else {
                            error!("Failed to get chat completion chunk: {:?}", err);
                        }
                        break;
                    }
                };

                if let Some(content) = chunk.choices[0].delta.content.as_deref() {
                    yield Ok(ThreadRunItem::ThreadAssistantMessageContentDelta(ThreadAssistantMessageContentDelta {
                        delta: content.to_owned()
                    }));
                }
            }
        };

        Ok(Box::pin(s))
    }

    async fn collect_relevant_code(
        &self,
        helper: &ContextInfoHelper,
        input: &CodeQueryInput,
        params: &CodeSearchParams,
        override_params: Option<&CodeSearchParamsOverrideInput>,
    ) -> Vec<CodeSearchHit> {
        let source_id: Option<&str> = {
            if let Some(source_id) = &input.source_id {
                // If source_id doesn't exist, return empty result.
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
        };

        let Some(source_id) = source_id else {
            return vec![];
        };

        let query = CodeSearchQuery::new(
            input.filepath.clone(),
            input.language.clone(),
            helper.rewrite_tag(&input.content),
            source_id.to_owned(),
        );

        let mut params = params.clone();
        override_params
            .as_ref()
            .inspect(|x| x.override_params(&mut params));

        match self.code.search_in_language(query, params).await {
            Ok(docs) => docs.hits,
            Err(err) => {
                if let CodeSearchError::NotReady = err {
                    debug!("Code search is not ready yet");
                } else {
                    warn!("Failed to search code: {:?}", err);
                }
                vec![]
            }
        }
    }

    async fn collect_relevant_docs(
        &self,
        helper: &ContextInfoHelper,
        doc_query: &DocQueryInput,
    ) -> Vec<DocSearchHit> {
        let mut source_ids = doc_query.source_ids.as_deref().unwrap_or_default().to_vec();

        // Only keep source_ids that are valid.
        source_ids.retain(|x| helper.can_access_source_id(x));

        // Rewrite [[source:${id}]] tags to the actual source name for doc search.
        let content = helper.rewrite_tag(&doc_query.content);

        let mut hits = vec![];

        // 1. Collect relevant docs from the tantivy doc search.
        if !source_ids.is_empty() {
            match self.doc.search(&source_ids, &content, 5).await {
                Ok(docs) => hits.extend(docs.hits),
                Err(err) => {
                    if let DocSearchError::NotReady = err {
                        debug!("Doc search is not ready yet");
                    } else {
                        warn!("Failed to search doc: {:?}", err);
                    }
                }
            };
        }

        // 2. If serper is available, we also collect from serper
        if doc_query.search_public {
            if let Some(serper) = self.serper.as_ref() {
                match serper.search(&[], &content, 5).await {
                    Ok(docs) => hits.extend(docs.hits),
                    Err(err) => {
                        warn!("Failed to search serper: {:?}", err);
                    }
                };
            }
        }

        hits
    }

    async fn generate_relevant_questions_v2(
        &self,
        attachment: &MessageAttachment,
        question: &str,
    ) -> Vec<String> {
        if attachment.code.is_empty() && attachment.doc.is_empty() {
            return vec![];
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
                    .map(|doc| format!("```\n{}\n```", doc.content)),
            )
            .collect();

        let context: String = snippets.join("\n\n");
        let prompt = format!(
            r#"
You are a helpful assistant that helps the user to ask related questions, based on user's original question and the related contexts. Please identify worthwhile topics that can be follow-ups, and write questions no longer than 20 words each. Please make sure that specifics, like events, names, locations, are included in follow up questions so they can be asked standalone. For example, if the original question asks about "the Manhattan project", in the follow up question, do not just say "the project", but use the full name "the Manhattan project". Your related questions must be in the same language as the original question.

Here are the contexts of the question:

{context}

Remember, based on the original question and related contexts, suggest three such further questions. Do NOT repeat the original question. Each related question should be no longer than 20 words. Here is the original question:

{question}
"#
        );

        let request = CreateChatCompletionRequestArgs::default()
            .messages(vec![ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(prompt)
                    .build()
                    .expect("Failed to create ChatCompletionRequestUserMessage"),
            )])
            .build()
            .expect("Failed to create ChatCompletionRequest");

        let chat = self.chat.clone();
        let s = chat
            .chat(request)
            .await
            .expect("Failed to create chat completion stream");
        let content = s.choices[0]
            .message
            .content
            .as_deref()
            .expect("Failed to get content from chat completion");
        content
            .lines()
            .map(trim_bullet)
            .filter(|x| !x.is_empty())
            .collect()
    }
}

fn trim_bullet(s: &str) -> String {
    let is_bullet = |c: char| c == '-' || c == '*' || c == '.' || c.is_numeric();
    s.trim()
        .trim_start_matches(is_bullet)
        .trim_end_matches(is_bullet)
        .trim()
        .to_owned()
}

pub fn create(
    config: &AnswerConfig,
    chat: Arc<dyn ChatCompletionStream>,
    code: Arc<dyn CodeSearch>,
    doc: Arc<dyn DocSearch>,
    context: Arc<dyn ContextService>,
    serper: Option<Box<dyn DocSearch>>,
) -> AnswerService {
    AnswerService::new(config, chat, code, doc, context, serper)
}

fn convert_messages_to_chat_completion_request(
    config: &AnswerConfig,
    helper: &ContextInfoHelper,
    messages: &[tabby_schema::thread::Message],
    attachment: &tabby_schema::thread::MessageAttachment,
    user_attachment_input: Option<&tabby_schema::thread::MessageAttachmentInput>,
) -> anyhow::Result<Vec<ChatCompletionRequestMessage>> {
    let mut output = vec![];
    output.reserve(messages.len() + 1);

    // System message
    output.push(ChatCompletionRequestMessage::System(
        ChatCompletionRequestSystemMessage {
            content: config.system_prompt.clone(),
            role: Role::System,
            name: None,
        },
    ));

    for i in 0..messages.len() - 1 {
        let x = &messages[i];
        let role = match x.role {
            thread::Role::Assistant => Role::Assistant,
            thread::Role::User => Role::User,
        };

        let content = if role == Role::User {
            if i % 2 != 0 {
                bail!("User message must be followed by assistant message");
            }

            let y = &messages[i + 1];

            build_user_prompt(&x.content, &y.attachment, None)
        } else {
            x.content.clone()
        };

        output.push(ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessage {
                content: helper.rewrite_tag(&content),
                role,
                name: None,
            },
        ));
    }

    output.push(ChatCompletionRequestMessage::System(
        ChatCompletionRequestSystemMessage {
            content: helper.rewrite_tag(&build_user_prompt(
                &messages[messages.len() - 1].content,
                attachment,
                user_attachment_input,
            )),
            role: Role::User,
            name: None,
        },
    ));

    Ok(output)
}

fn build_user_prompt(
    user_input: &str,
    assistant_attachment: &tabby_schema::thread::MessageAttachment,
    user_attachment_input: Option<&tabby_schema::thread::MessageAttachmentInput>,
) -> String {
    // If the user message has no code attachment and the assistant message has no code attachment or doc attachment, return the user message directly.
    if user_attachment_input
        .map(|x| x.code.is_empty())
        .unwrap_or(true)
        && assistant_attachment.code.is_empty()
        && assistant_attachment.doc.is_empty()
    {
        return user_input.to_owned();
    }

    let snippets: Vec<String> = assistant_attachment
        .doc
        .iter()
        .map(|doc| format!("```\n{}\n```", doc.content))
        .chain(
            user_attachment_input
                .map(|x| &x.code)
                .unwrap_or(&vec![])
                .iter()
                .map(|snippet| {
                    if let Some(filepath) = &snippet.filepath {
                        format!("```title=\"{}\"\n{}\n```", filepath, snippet.content)
                    } else {
                        format!("```\n{}\n```", snippet.content)
                    }
                }),
        )
        .chain(assistant_attachment.code.iter().map(|snippet| {
            format!(
                "```{} title=\"{}\"\n{}\n```",
                snippet.language, snippet.filepath, snippet.content
            )
        }))
        .collect();

    let citations: Vec<String> = snippets
        .iter()
        .enumerate()
        .map(|(i, snippet)| format!("[[citation:{}]]\n{}", i + 1, *snippet))
        .collect();

    let context = citations.join("\n\n");

    format!(
        r#"You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable.

Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

Please cite the contexts with the reference numbers, in the format [[citation:x]]. If a sentence comes from multiple contexts, please list all applicable citations, like [[citation:3]][[citation:5]]. Other than code and specific names and citations, your answer must be written in the same language as the question.

Here are the set of contexts:

{context}

Remember, don't blindly repeat the contexts verbatim. When possible, give code snippet to demonstrate the answer. And here is the user question:

{user_input}
"#
    )
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
            doc::DocSearch,
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

    use crate::answer::{
        testutils::{
            FakeChatCompletionStream, FakeCodeSearch, FakeCodeSearchFail,
            FakeCodeSearchFailNotReady, FakeContextService, FakeDocSearch,
        },
        trim_bullet, AnswerService,
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

    pub fn make_context_info_helper() -> ContextInfoHelper {
        ContextInfoHelper::new(&ContextInfo {
            sources: vec![ContextSourceValue::Repository(Repository {
                id: ID::from(TEST_SOURCE_ID.to_owned()),
                source_id: TEST_SOURCE_ID.to_owned(),
                name: "tabby".to_owned(),
                kind: RepositoryKind::Github,
                dir: PathBuf::from("tabby"),
                git_url: TEST_GIT_URL.to_owned(),
                refs: vec![],
            })],
        })
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
            doc: vec![tabby_schema::thread::MessageAttachmentDoc {
                title: "Documentation".to_owned(),
                content: "This code implements a basic web server.".to_owned(),
                link: "https://example.com/docs".to_owned(),
            }],
            code: vec![tabby_schema::thread::MessageAttachmentCode {
                git_url: "https://github.com/".to_owned(),
                filepath: "server.py".to_owned(),
                language: "python".to_owned(),
                content: "from flask import Flask\n\napp = Flask(__name__)\n\n@app.route('/')\ndef hello():\n    return 'Hello, World!'".to_owned(),
                start_line: 1,
            }],
            client_code: vec![],
        };
        let user_attachment_input = None;

        let prompt =
            super::build_user_prompt(user_input, &assistant_attachment, user_attachment_input);

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
            doc: vec![tabby_schema::thread::MessageAttachmentDoc {
                title: "1. Example Document".to_owned(),
                content: "This is an example".to_owned(),
                link: "https://example.com".to_owned(),
            }],
            code: vec![tabby_schema::thread::MessageAttachmentCode {
                git_url: "https://github.com".to_owned(),
                filepath: "server.py".to_owned(),
                language: "python".to_owned(),
                content: "print('Hello, server!')".to_owned(),
                start_line: 1,
            }],
            client_code: vec![tabby_schema::thread::MessageAttachmentClientCode {
                filepath: Some("client.py".to_owned()),
                content: "print('Hello, client!')".to_owned(),
                start_line: Some(1),
            }],
        };

        let messages = vec![
            make_message(1, "Hello", tabby_schema::thread::Role::User, None),
            make_message(
                2,
                "Hi, [[source:preset_web_document:source-1]], [[source:2]]",
                tabby_schema::thread::Role::Assistant,
                Some(attachment),
            ),
            make_message(3, "How are you?", tabby_schema::thread::Role::User, None),
        ];

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
        let output = super::convert_messages_to_chat_completion_request(
            &config,
            &rewriter,
            &messages,
            &tabby_schema::thread::MessageAttachment::default(),
            Some(&user_attachment_input),
        )
        .unwrap();

        insta::assert_yaml_snapshot!(output);
    }

    #[tokio::test]
    async fn test_collect_relevant_code() {
        let chat: Arc<dyn ChatCompletionStream> = Arc::new(FakeChatCompletionStream);
        let code: Arc<dyn CodeSearch> = Arc::new(FakeCodeSearch);
        let doc: Arc<dyn DocSearch> = Arc::new(FakeDocSearch);
        let context: Arc<dyn ContextService> = Arc::new(FakeContextService);
        let mut serper = Some(Box::new(FakeDocSearch) as Box<dyn DocSearch>);
        let config = make_answer_config();
        let mut service = AnswerService::new(
            &config,
            chat.clone(),
            code.clone(),
            doc.clone(),
            context.clone(),
            serper,
        );
        let code_query_input_could_access =
            make_code_query_input(Some(TEST_SOURCE_ID), Some(TEST_GIT_URL));
        let code_search_params = make_code_search_params();
        let context_info_helper: ContextInfoHelper = make_context_info_helper();
        debug_assert!(context_info_helper.can_access_source_id("source-1"));

        service
            .collect_relevant_code(
                &context_info_helper,
                &code_query_input_could_access,
                &code_search_params,
                None,
            )
            .await;

        let code_query_input_not_access = make_code_query_input(Some("TEST"), Some(TEST_GIT_URL));
        service
            .collect_relevant_code(
                &context_info_helper,
                &code_query_input_not_access,
                &code_search_params,
                None,
            )
            .await;

        let code_query_input_with_only_git = make_code_query_input(None, Some(TEST_GIT_URL));
        service
            .collect_relevant_code(
                &context_info_helper,
                &code_query_input_with_only_git,
                &code_search_params,
                None,
            )
            .await;

        let code_query_input_with_only_git = make_code_query_input(None, None);
        service
            .collect_relevant_code(
                &context_info_helper,
                &code_query_input_with_only_git,
                &code_search_params,
                None,
            )
            .await;

        let code_fail_not_ready = Arc::new(FakeCodeSearchFailNotReady);
        serper = Some(Box::new(FakeDocSearch) as Box<dyn DocSearch>);

        service = AnswerService::new(
            &config,
            chat.clone(),
            code_fail_not_ready.clone(),
            doc.clone(),
            context.clone(),
            serper,
        );

        let code_fail = Arc::new(FakeCodeSearchFail);
        serper = Some(Box::new(FakeDocSearch) as Box<dyn DocSearch>);

        service = AnswerService::new(
            &config,
            chat.clone(),
            code_fail.clone(),
            doc.clone(),
            context.clone(),
            serper,
        )
    }

    #[tokio::test]
    async fn test_generate_relevant_questions_v2() {
        let chat: Arc<dyn ChatCompletionStream> = Arc::new(FakeChatCompletionStream);
        let code: Arc<dyn CodeSearch> = Arc::new(FakeCodeSearch);
        let doc: Arc<dyn DocSearch> = Arc::new(FakeDocSearch);
        let context: Arc<dyn ContextService> = Arc::new(FakeContextService);
        let serper = Some(Box::new(FakeDocSearch) as Box<dyn DocSearch>);
        let config = make_answer_config();
        let service = AnswerService::new(
            &config,
            chat.clone(),
            code.clone(),
            doc.clone(),
            context.clone(),
            serper,
        );

        let attachment = MessageAttachment {
            doc: vec![tabby_schema::thread::MessageAttachmentDoc {
                title: "1. Example Document".to_owned(),
                content: "This is an example".to_owned(),
                link: "https://example.com".to_owned(),
            }],
            code: vec![tabby_schema::thread::MessageAttachmentCode {
                git_url: "https://github.com".to_owned(),
                filepath: "server.py".to_owned(),
                language: "python".to_owned(),
                content: "print('Hello, server!')".to_owned(),
                start_line: 1,
            }],
            client_code: vec![tabby_schema::thread::MessageAttachmentClientCode {
                filepath: Some("client.py".to_owned()),
                content: "print('Hello, client!')".to_owned(),
                start_line: Some(1),
            }],
        };

        let question = "What is the purpose of this code?";

        let result = service
            .generate_relevant_questions_v2(&attachment, question)
            .await;

        let expected = vec![
            "What is the main functionality of the provided code?".to_string(),
            "How does the code snippet implement a web server?".to_string(),
            "Can you explain how the Flask app works in this context?".to_string(),
        ];

        assert_eq!(result, expected);
    }
    #[tokio::test]
    async fn test_collect_relevant_docs() {
        let chat: Arc<dyn ChatCompletionStream> = Arc::new(FakeChatCompletionStream);
        let code: Arc<dyn CodeSearch> = Arc::new(FakeCodeSearch);
        let doc: Arc<dyn DocSearch> = Arc::new(FakeDocSearch);
        let context: Arc<dyn ContextService> = Arc::new(FakeContextService);
        let serper = Some(Box::new(FakeDocSearch) as Box<dyn DocSearch>);
        let config = make_answer_config();
        let service = AnswerService::new(
            &config,
            chat.clone(),
            code.clone(),
            doc.clone(),
            context.clone(),
            serper,
        );

        let context_info_helper = make_context_info_helper();
        let doc_query = tabby_schema::thread::DocQueryInput {
            content: "Test query Here[[source:source-1]]".to_string(),
            source_ids: Some(vec!["source-1".to_string()]),
            search_public: true,
        };

        let hits = service
            .collect_relevant_docs(&context_info_helper, &doc_query)
            .await;

        assert_eq!(hits.len(), 10, "Expected 10 hits from the doc search");

        assert!(
            hits.iter().any(|hit| hit.doc.title == "Document 1"),
            "Expected to find a hit with title 'Document 1'"
        );
    }

    #[test]
    fn test_trim_bullet() {
        assert_eq!(trim_bullet("- Hello"), "Hello");
        assert_eq!(trim_bullet("* World"), "World");
        assert_eq!(trim_bullet("1. Test"), "Test");
        assert_eq!(trim_bullet(".Dot"), "Dot");

        assert_eq!(trim_bullet("- Hello -"), "Hello");
        assert_eq!(trim_bullet("1. Test 1"), "Test");

        assert_eq!(trim_bullet("--** Mixed"), "Mixed");

        assert_eq!(trim_bullet("  - Hello  "), "Hello");

        assert_eq!(trim_bullet("-"), "");
        assert_eq!(trim_bullet(""), "");
        assert_eq!(trim_bullet("   "), "");

        assert_eq!(trim_bullet("Hello World"), "Hello World");

        assert_eq!(trim_bullet("1. *Bold* and -italic-"), "*Bold* and -italic");
    }
    #[tokio::test]
    async fn test_answer_v2() {
        use std::sync::Arc;

        use futures::StreamExt;
        use tabby_schema::{policy::AccessPolicy, thread::ThreadRunOptionsInput};

        let chat: Arc<dyn ChatCompletionStream> = Arc::new(FakeChatCompletionStream);
        let code: Arc<dyn CodeSearch> = Arc::new(FakeCodeSearch);
        let doc: Arc<dyn DocSearch> = Arc::new(FakeDocSearch);
        let context: Arc<dyn ContextService> = Arc::new(FakeContextService);
        let serper = Some(Box::new(FakeDocSearch) as Box<dyn DocSearch>);

        let config = make_answer_config();
        let service = Arc::new(AnswerService::new(
            &config, chat, code, doc, context, serper,
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
            .answer_v2(&policy, &messages, &options, user_attachment_input)
            .await
            .unwrap();

        let collected_results: Vec<_> = result.collect().await;

        assert_eq!(
            collected_results.len(),
            4,
            "Expected 4 items in the result stream"
        );
    }
}
