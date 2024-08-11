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
use tabby_common::api::{
    answer::{AnswerRequest, AnswerResponseChunk},
    code::{CodeSearch, CodeSearchError, CodeSearchHit, CodeSearchQuery},
    doc::{DocSearch, DocSearchError, DocSearchHit},
};
use tabby_inference::ChatCompletionStream;
use tabby_schema::{
    repository::RepositoryService,
    thread::{
        self, CodeQueryInput, DocQueryInput, MessageAttachmentCode, ThreadRunItem,
        ThreadRunOptionsInput,
    },
    web_crawler::WebCrawlerService,
};
use tracing::{debug, error, warn};

pub struct AnswerService {
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
            chat,
            code,
            doc,
            web,
            repository,
            serper,
        }
    }

    #[deprecated(note = "This shall be removed after the migration to v2 is done.")]
    pub async fn answer<'a>(
        self: Arc<Self>,
        mut req: AnswerRequest,
    ) -> BoxStream<'a, AnswerResponseChunk> {
        let s = stream! {
            // 0. Collect sources given query, for now we only use the last message
            let query: &mut _ = match req.messages.last_mut() {
                Some(query) => query,
                None => {
                    warn!("No query found in the request");
                    return;
                }
            };

            let git_url = req.code_query.as_ref().map(|x| x.git_url.clone());

            // 0. Extract client-provided code snippets
            let code_snippets = req.code_snippets;

            // 1. Collect relevant code if needed.
            let relevant_code = if let Some(mut code_query)  = req.code_query  {
                if req.collect_relevant_code_using_user_message {
                    // Natural language content from query is used to search for relevant code.
                    code_query.content = get_content(query).to_owned();
                } else {
                    // Code snippet is extended to the query.
                    self.override_query_with_code_query(query, &code_query).await;
                }

                let code_query = CodeQueryInput {
                    git_url: code_query.git_url,
                    filepath: code_query.filepath,
                    language: code_query.language,
                    content: code_query.content,
                };
                self.collect_relevant_code(&code_query).await
            } else {
                vec![]
            };

            if !relevant_code.is_empty() {
                yield AnswerResponseChunk::RelevantCode(relevant_code.clone());
            }


            // 2. Collect relevant docs if needed.
            let relevant_docs = if req.doc_query {
                let query = DocQueryInput {
                    content: get_content(query).to_owned(),
                };
                self.collect_relevant_docs(git_url.as_deref(), &query).await
            } else {
                vec![]
            };

            if !relevant_docs.is_empty() {
                yield AnswerResponseChunk::RelevantDocuments(relevant_docs.clone());
            }

            if !code_snippets.is_empty() || !relevant_code.is_empty() || !relevant_docs.is_empty() {
                if req.generate_relevant_questions {
                    // 3. Generate relevant questions from the query
                    let relevant_questions = self.generate_relevant_questions(&relevant_code, &relevant_docs, get_content(query)).await;
                    yield AnswerResponseChunk::RelevantQuestions(relevant_questions);
                }

                let code_snippets: Vec<MessageAttachmentCode> = code_snippets.iter().map(|x| MessageAttachmentCode {
                    filepath: x.filepath.clone(),
                    content: x.content.clone(),
                }).collect();

                // 4. Generate override prompt from the query
                set_content(query, self.generate_prompt(&code_snippets, &relevant_code, &relevant_docs, get_content(query)).await);
            }


            // 5. Generate answer from the query
            let request = {
                let mut builder = CreateChatCompletionRequestArgs::default();
                builder.messages(req.messages).presence_penalty(PRESENCE_PENALTY);
                if let Some(user) = req.user {
                    builder.user(user);
                };

                builder.build().expect("Failed to create ChatCompletionRequest")
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
                        debug!("Failed to get chat completion chunk: {:?}", err);
                        break;
                    }
                };

                if let Some(content) = chunk.choices[0].delta.content.as_deref() {
                    yield AnswerResponseChunk::AnswerDelta(content.to_owned());
                }
            }
        };

        Box::pin(s)
    }

    pub async fn answer_v2<'a>(
        self: Arc<Self>,
        messages: &[tabby_schema::thread::Message],
        options: &ThreadRunOptionsInput,
    ) -> tabby_schema::Result<BoxStream<'a, tabby_schema::Result<ThreadRunItem>>> {
        let messages = messages.to_vec();
        let options = options.clone();

        let s = stream! {
            let query = match messages.last() {
                Some(query) => query,
                None => {
                    yield Err(anyhow!("No query found in the request").into());
                    return;
                }
            };

            let git_url = options.code_query.as_ref().map(|x| x.git_url.clone());

            // 1. Collect relevant code if needed.
            let relevant_code = if let Some(code_query) = options.code_query.as_ref() {
                self.collect_relevant_code(code_query).await
            } else {
                vec![]
            };

            relevant_code.is_empty();

            // 2. Collect relevant docs if needed.
            let relevant_docs = if let Some(doc_query) = options.doc_query.as_ref() {
                self.collect_relevant_docs(git_url.as_deref(), doc_query)
                    .await
            } else {
                vec![]
            };

            if !relevant_docs.is_empty() {
                yield Ok(ThreadRunItem::ThreadAssistantMessageAttachmentsCode(
                    relevant_code
                        .iter()
                        .map(|x| MessageAttachmentCode {
                            filepath: Some(x.doc.filepath.clone()),
                            content: x.doc.body.clone(),
                        })
                        .collect::<Vec<_>>(),
                ));
            }

            // 3. Generate relevant questions.
            if options.generate_relevant_questions {
                let questions = self
                    .generate_relevant_questions(&relevant_code, &relevant_docs, &query.content)
                    .await;
                yield Ok(ThreadRunItem::ThreadRelevantQuestions(questions));
            }

            // 4. Prepare requesting LLM
            let request = {
                let code_snippets: &[MessageAttachmentCode] = &query
                    .attachment
                    .code;

                let override_user_prompt = if !code_snippets.is_empty()
                    || !relevant_code.is_empty()
                    || !relevant_docs.is_empty()
                {
                    self.generate_prompt(
                        code_snippets,
                        &relevant_code,
                        &relevant_docs,
                        &query.content,
                    )
                    .await
                } else {
                    query.content.clone()
                };

                // Convert `messages` to CreateChatCompletionRequest
                let chat_messages: Vec<_> = messages
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        let role = match x.role {
                            thread::Role::Assistant => Role::Assistant,
                            thread::Role::User => Role::User,
                        };

                        let is_last = i == messages.len() - 1;
                        let content = if is_last {
                            override_user_prompt.clone()
                        } else {
                            x.content.clone()
                        };

                        ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                            content,
                            role,
                            name: None,
                        })
                    })
                    .collect();

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

    async fn collect_relevant_code(&self, query: &CodeQueryInput) -> Vec<CodeSearchHit> {
        let query = CodeSearchQuery {
            git_url: query.git_url.clone(),
            filepath: query.filepath.clone(),
            language: query.language.clone(),
            content: query.content.clone(),
        };
        match self.code.search_in_language(query, 20).await {
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
                if let Ok(git_source_id) = self
                    .repository
                    .resolve_web_source_id_by_git_url(git_url)
                    .await
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

    async fn generate_relevant_questions(
        &self,
        relevant_code: &[CodeSearchHit],
        relevant_docs: &[DocSearchHit],
        question: &str,
    ) -> Vec<String> {
        let snippets: Vec<String> = relevant_code
            .iter()
            .map(|hit| {
                format!(
                    "```{} title=\"{}\"\n{}\n```",
                    hit.doc.language, hit.doc.filepath, hit.doc.body
                )
            })
            .chain(relevant_docs.iter().map(|hit| hit.doc.snippet.to_owned()))
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

    async fn override_query_with_code_query(
        &self,
        query: &mut ChatCompletionRequestMessage,
        code_query: &CodeSearchQuery,
    ) {
        set_content(
            query,
            format!(
                "{}\n\n```{}\n{}\n```",
                get_content(query),
                code_query.language.as_deref().unwrap_or_default(),
                code_query.content
            ),
        )
    }

    async fn generate_prompt(
        &self,
        code_snippets: &[MessageAttachmentCode],
        relevant_code: &[CodeSearchHit],
        relevant_docs: &[DocSearchHit],
        question: &str,
    ) -> String {
        let snippets: Vec<String> = code_snippets
            .iter()
            .map(|snippet| {
                if let Some(filepath) = &snippet.filepath {
                    format!("```title=\"{}\"\n{}\n```", filepath, snippet.content)
                } else {
                    format!("```\n{}\n```", snippet.content)
                }
            })
            .chain(relevant_code.iter().map(|hit| {
                format!(
                    "```{} title=\"{}\"\n{}\n```",
                    hit.doc.language, hit.doc.filepath, hit.doc.body
                )
            }))
            .chain(relevant_docs.iter().map(|hit| hit.doc.snippet.to_owned()))
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

{question}
"#
        )
    }
}

fn remove_bullet_prefix(s: &str) -> String {
    s.trim()
        .trim_start_matches(|c: char| c == '-' || c == '*' || c == '.' || c.is_numeric())
        .trim()
        .to_owned()
}

pub fn create(
    chat: Arc<dyn ChatCompletionStream>,
    code: Arc<dyn CodeSearch>,
    doc: Arc<dyn DocSearch>,
    web: Arc<dyn WebCrawlerService>,
    repository: Arc<dyn RepositoryService>,
    serper_factory_fn: impl Fn(&str) -> Box<dyn DocSearch>,
) -> AnswerService {
    AnswerService::new(chat, code, doc, web, repository, serper_factory_fn)
}

fn get_content(message: &ChatCompletionRequestMessage) -> &str {
    match message {
        ChatCompletionRequestMessage::System(x) => &x.content,
        _ => {
            panic!("Unexpected message type, {:?}", message);
        }
    }
}

fn set_content(message: &mut ChatCompletionRequestMessage, content: String) {
    match message {
        ChatCompletionRequestMessage::System(x) => x.content = content,
        _ => {
            panic!("Unexpected message type");
        }
    }
}
