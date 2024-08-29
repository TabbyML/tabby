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
    repository::RepositoryService,
    thread::{
        self, CodeQueryInput, CodeSearchParamsOverrideInput, DocQueryInput, MessageAttachment,
        ThreadRunItem, ThreadRunOptionsInput,
    },
    web_crawler::WebCrawlerService,
};
use tracing::{debug, error, warn};

use crate::bail;

pub struct AnswerService {
    config: AnswerConfig,
    chat: Arc<dyn ChatCompletionStream>,
    code: Arc<dyn CodeSearch>,
    doc: Arc<dyn DocSearch>,
    web: Arc<dyn WebCrawlerService>,
    repository: Arc<dyn RepositoryService>,
    serper: Option<Box<dyn DocSearch>>,
}

// FIXME(meng): make this configurable.
const PRESENCE_PENALTY: f32 = 0.5;

impl AnswerService {
    fn new(
        config: &AnswerConfig,
        chat: Arc<dyn ChatCompletionStream>,
        code: Arc<dyn CodeSearch>,
        doc: Arc<dyn DocSearch>,
        web: Arc<dyn WebCrawlerService>,
        repository: Arc<dyn RepositoryService>,
        serper_factory_fn: impl Fn(&str) -> Box<dyn DocSearch>,
    ) -> Self {
        let serper: Option<Box<dyn DocSearch>> =
            if let Ok(api_key) = std::env::var("SERPER_API_KEY") {
                debug!("Serper API key found, enabling serper...");
                Some(serper_factory_fn(&api_key))
            } else {
                None
            };
        Self {
            config: config.clone(),
            chat,
            code,
            doc,
            web,
            repository,
            serper,
        }
    }

    pub async fn answer_v2<'a>(
        self: Arc<Self>,
        messages: &[tabby_schema::thread::Message],
        options: &ThreadRunOptionsInput,
        user_attachment_input: Option<&tabby_schema::thread::MessageAttachmentInput>,
    ) -> tabby_schema::Result<BoxStream<'a, tabby_schema::Result<ThreadRunItem>>> {
        let messages = messages.to_vec();
        let options = options.clone();
        let user_attachment_input = user_attachment_input.cloned();

        let s = stream! {
            let query = match messages.last() {
                Some(query) => query,
                None => {
                    yield Err(anyhow!("No query found in the request").into());
                    return;
                }
            };

            let git_url = options.code_query.as_ref().map(|x| x.git_url.clone());
            let mut attachment = MessageAttachment::default();

            // 1. Collect relevant code if needed.
            if let Some(code_query) = options.code_query.as_ref() {
                let hits = self.collect_relevant_code(code_query, &self.config.code_search_params, options.debug_options.as_ref().and_then(|x| x.code_search_params_override.as_ref())).await;
                attachment.code = hits.iter().map(|x| x.doc.clone().into()).collect::<Vec<_>>();

                if !hits.is_empty() {
                    let message_hits = hits.into_iter().map(|x| x.into()).collect::<Vec<_>>();
                    yield Ok(ThreadRunItem::ThreadAssistantMessageAttachmentsCode(
                        message_hits
                    ));
                }
            };

            // 2. Collect relevant docs if needed.
            if let Some(doc_query) = options.doc_query.as_ref() {
                let hits = self.collect_relevant_docs(git_url.as_deref(), doc_query)
                    .await;
                attachment.doc = hits.iter()
                        .map(|x| x.doc.clone().into())
                        .collect::<Vec<_>>();

                if !attachment.doc.is_empty() {
                    let message_hits = hits.into_iter().map(|x| x.into()).collect::<Vec<_>>();
                    yield Ok(ThreadRunItem::ThreadAssistantMessageAttachmentsDoc(
                        message_hits
                    ));
                }
            };

            // 3. Generate relevant questions.
            if options.generate_relevant_questions {
                let questions = self
                    .generate_relevant_questions_v2(&attachment, &query.content)
                    .await;
                yield Ok(ThreadRunItem::ThreadRelevantQuestions(questions));
            }

            // 4. Prepare requesting LLM
            let request = {
                let chat_messages = convert_messages_to_chat_completion_request(&messages, &attachment, user_attachment_input.as_ref())?;

                CreateChatCompletionRequestArgs::default()
                    .messages(chat_messages)
                    .presence_penalty(PRESENCE_PENALTY)
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
                    yield Ok(ThreadRunItem::ThreadAssistantMessageContentDelta(content.to_owned()));
                }
            }
        };

        Ok(Box::pin(s))
    }

    async fn collect_relevant_code(
        &self,
        input: &CodeQueryInput,
        params: &CodeSearchParams,
        override_params: Option<&CodeSearchParamsOverrideInput>,
    ) -> Vec<CodeSearchHit> {
        let query = CodeSearchQuery::new(
            input.git_url.clone(),
            input.filepath.clone(),
            input.language.clone(),
            input.content.clone(),
        );

        let mut params = params.clone();
        override_params
            .as_ref()
            .inspect(|x| x.override_params(&mut params));

        match self.code.search_in_language(query, params.clone()).await {
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
        code_query_git_url: Option<&str>,
        doc_query: &DocQueryInput,
    ) -> Vec<DocSearchHit> {
        let source_ids = {
            // 1. By default only web sources are considered.
            let mut source_ids: Vec<_> = self
                .web
                .list_web_crawler_urls(None, None, None, None)
                .await
                .unwrap_or_default()
                .into_iter()
                .map(|url| url.source_id())
                .collect();

            // 2. If code_query is available, we also issues / PRs coming from the source.
            if let Some(git_url) = code_query_git_url {
                if let Ok(git_source_id) =
                    self.repository.resolve_source_id_by_git_url(git_url).await
                {
                    source_ids.push(git_source_id);
                }
            }

            source_ids
        };

        if source_ids.is_empty() {
            return vec![];
        }

        // 1. Collect relevant docs from the tantivy doc search.
        let mut hits = vec![];
        let doc_hits = match self.doc.search(&source_ids, &doc_query.content, 5).await {
            Ok(docs) => docs.hits,
            Err(err) => {
                if let DocSearchError::NotReady = err {
                    debug!("Doc search is not ready yet");
                } else {
                    warn!("Failed to search doc: {:?}", err);
                }
                vec![]
            }
        };
        hits.extend(doc_hits);

        // 2. If serper is available, we also collect from serper
        if let Some(serper) = self.serper.as_ref() {
            let serper_hits = match serper.search(&[], &doc_query.content, 5).await {
                Ok(docs) => docs.hits,
                Err(err) => {
                    warn!("Failed to search serper: {:?}", err);
                    vec![]
                }
            };
            hits.extend(serper_hits);
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
            .map(remove_bullet_prefix)
            .filter(|x| !x.is_empty())
            .collect()
    }

    pub fn can_search_public(&self) -> bool {
        self.serper.is_some()
    }
}

fn remove_bullet_prefix(s: &str) -> String {
    s.trim()
        .trim_start_matches(|c: char| c == '-' || c == '*' || c == '.' || c.is_numeric())
        .trim()
        .to_owned()
}

pub fn create(
    config: &AnswerConfig,
    chat: Arc<dyn ChatCompletionStream>,
    code: Arc<dyn CodeSearch>,
    doc: Arc<dyn DocSearch>,
    web: Arc<dyn WebCrawlerService>,
    repository: Arc<dyn RepositoryService>,
    serper_factory_fn: impl Fn(&str) -> Box<dyn DocSearch>,
) -> AnswerService {
    AnswerService::new(config, chat, code, doc, web, repository, serper_factory_fn)
}

fn convert_messages_to_chat_completion_request(
    messages: &[tabby_schema::thread::Message],
    attachment: &tabby_schema::thread::MessageAttachment,
    user_attachment_input: Option<&tabby_schema::thread::MessageAttachmentInput>,
) -> anyhow::Result<Vec<ChatCompletionRequestMessage>> {
    let mut output = vec![];
    output.reserve(messages.len());

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
                content,
                role,
                name: None,
            },
        ));
    }

    output.push(ChatCompletionRequestMessage::System(
        ChatCompletionRequestSystemMessage {
            content: build_user_prompt(
                &messages[messages.len() - 1].content,
                attachment,
                user_attachment_input,
            ),
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
        r#"
You are a professional developer AI assistant. You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable.

Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

Please cite the contexts with the reference numbers, in the format [citation:x]. If a sentence comes from multiple contexts, please list all applicable citations, like [citation:3][citation:5]. Other than code and specific names and citations, your answer must be written in the same language as the question.

Here are the set of contexts:

{context}

Remember, don't blindly repeat the contexts verbatim. When possible, give code snippet to demonstrate the answer. And here is the user question:

{user_input}
"#
    )
}

#[cfg(test)]
mod tests {
    use juniper::ID;
    use tabby_schema::AsID;

    fn make_message(
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
                "Hi",
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

        let output = super::convert_messages_to_chat_completion_request(
            &messages,
            &tabby_schema::thread::MessageAttachment::default(),
            Some(&user_attachment_input),
        )
        .unwrap();

        insta::assert_yaml_snapshot!(output);
    }
}
