mod prompt_tools;

use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::anyhow;
use async_openai_alt::{
    error::OpenAIError,
    types::{
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, CreateChatCompletionRequestArgs, Role,
    },
};
use async_stream::stream;
use futures::stream::BoxStream;
use prompt_tools::{pipeline_decide_need_codebase_context, pipeline_related_questions};
use tabby_common::{
    api::{
        code::{
            CodeSearch, CodeSearchError, CodeSearchHit, CodeSearchParams, CodeSearchQuery,
            CodeSearchScores,
        },
        commit::{CommitHistorySearch, CommitHistorySearchHit, CommitHistorySearchParams},
        structured_doc::{DocSearch, DocSearchDocument, DocSearchError, DocSearchHit},
        SearchError,
    },
    config::AnswerConfig,
};
use tabby_inference::ChatCompletionStream;
use tabby_schema::{
    auth::AuthenticationService,
    context::{ContextInfoHelper, ContextService},
    policy::AccessPolicy,
    repository::{Repository, RepositoryService},
    thread::{
        self, CodeQueryInput, CodeSearchParamsOverrideInput, DocQueryInput, MessageAttachment,
        MessageAttachmentCodeFileList, MessageAttachmentCodeInput, MessageAttachmentDoc,
        MessageAttachmentInput, MessageDocSearchHit, ThreadAssistantMessageAttachmentsCode,
        ThreadAssistantMessageAttachmentsCodeFileList, ThreadAssistantMessageAttachmentsCommit,
        ThreadAssistantMessageAttachmentsDoc, ThreadAssistantMessageContentDelta,
        ThreadRelevantQuestions, ThreadRunItem, ThreadRunOptionsInput,
    },
};
use tracing::{debug, error, warn};

use crate::bail;

pub struct AnswerService {
    config: AnswerConfig,
    auth: Arc<dyn AuthenticationService>,
    chat: Arc<dyn ChatCompletionStream>,
    code: Arc<dyn CodeSearch>,
    doc: Arc<dyn DocSearch>,
    commit: Arc<dyn CommitHistorySearch>,
    context: Arc<dyn ContextService>,
    serper: Option<Box<dyn DocSearch>>,
    repository: Arc<dyn RepositoryService>,
}

impl AnswerService {
    fn new(
        config: &AnswerConfig,
        auth: Arc<dyn AuthenticationService>,
        chat: Arc<dyn ChatCompletionStream>,
        code: Arc<dyn CodeSearch>,
        doc: Arc<dyn DocSearch>,
        commit: Arc<dyn CommitHistorySearch>,
        context: Arc<dyn ContextService>,
        serper: Option<Box<dyn DocSearch>>,
        repository: Arc<dyn RepositoryService>,
    ) -> Self {
        Self {
            config: config.clone(),
            auth,
            chat,
            code,
            doc,
            commit,
            context,
            serper,
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
                if let Some(repository) = self.find_repository(&context_info_helper, code_query, policy.clone()).await {
                    let need_codebase_context = pipeline_decide_need_codebase_context(self.chat.clone(), &query.content).await?;
                    yield Ok(ThreadRunItem::ThreadAssistantMessageReadingCode(need_codebase_context.clone()));
                    if need_codebase_context.file_list {
                        // List at most 300 files in the repository.
                        match self.repository.list_files(&policy, &repository.kind, &repository.id, None, Some(300)).await {
                            Ok((files, _is_clipped)) => {
                                let file_list: Vec<_> = files.into_iter().map(|x| x.path).collect();
                                attachment.code_file_list = Some(MessageAttachmentCodeFileList {
                                    file_list: file_list.clone(),
                                });
                                yield Ok(ThreadRunItem::ThreadAssistantMessageAttachmentsCodeFileList(ThreadAssistantMessageAttachmentsCodeFileList {
                                    file_list,
                                }));
                            }
                            Err(e) => {
                                error!("failed to list files for repository {}: {}", repository.id, e);
                            }
                        }
                    }

                    if need_codebase_context.snippet {
                        let hits = self.collect_relevant_code(
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
                                ThreadAssistantMessageAttachmentsCode { code_source_id: repository.source_id.clone(), hits }
                            ));
                        }
                    }

                    if need_codebase_context.commit_history {
                        let params = CommitHistorySearchParams::default();
                        let hits = self.collect_commit_history(
                            &repository,
                            &context_info_helper,
                            code_query,
                            &params,
                        ).await;
                        attachment.commit = hits.iter().map(|x| x.commit.clone().into()).collect::<Vec<_>>();

                        if !hits.is_empty() {
                            let hits = hits.into_iter().map(|x| x.into()).collect::<Vec<_>>();
                            yield Ok(ThreadRunItem::ThreadAssistantMessageAttachmentsCommit(
                                ThreadAssistantMessageAttachmentsCommit { hits }
                            ));
                        }
                    }
                };
            };

            // 2. Collect relevant docs if needed.
            if let Some(doc_query) = options.doc_query.as_ref() {
                let hits = self.collect_relevant_docs(&context_info_helper, doc_query)
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
                let content = context_info_helper.rewrite_tag(&query.content);
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
                let chat_messages = convert_messages_to_chat_completion_request(&self.config, &context_info_helper, &messages, &attachment, user_attachment_input.as_ref())?;

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
        let email = match &doc {
            DocSearchDocument::Issue(issue) => issue.author_email.as_deref(),
            DocSearchDocument::Pull(pull) => pull.author_email.as_deref(),
            _ => None,
        };
        let user = if let Some(email) = email {
            auth.get_user_by_email(email).await.ok().map(|x| x.into())
        } else {
            None
        };
        MessageAttachmentDoc::from_doc_search_document(doc, user)
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

    async fn collect_relevant_code(
        &self,
        repository: &Repository,
        helper: &ContextInfoHelper,
        input: &CodeQueryInput,
        params: &CodeSearchParams,
        override_params: Option<&CodeSearchParamsOverrideInput>,
    ) -> Vec<CodeSearchHit> {
        let query = CodeSearchQuery::new(
            input.filepath.clone(),
            input.language.clone(),
            helper.rewrite_tag(&input.content),
            repository.source_id.clone(),
        );

        let mut params = params.clone();
        if let Some(override_params) = override_params {
            override_params.override_params(&mut params);
        }

        match self.code.search_in_language(query, params).await {
            Ok(docs) => merge_code_snippets(repository, docs.hits).await,
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

    async fn collect_commit_history(
        &self,
        repository: &Repository,
        helper: &ContextInfoHelper,
        code_query: &CodeQueryInput,
        params: &CommitHistorySearchParams,
    ) -> Vec<CommitHistorySearchHit> {
        if let Some(source_id) = &code_query.source_id {
            if !helper.can_access_source_id(source_id) {
                return vec![];
            }
        }

        // Rewrite [[source:${id}]] tags to the actual source name for doc search.
        let content = helper.rewrite_tag(&code_query.content);

        match self
            .commit
            .search(&repository.source_id, &content, params)
            .await
        {
            Ok(commits) => commits.hits,
            Err(err) => {
                if let SearchError::NotReady = err {
                    debug!("Commit search is not ready yet");
                } else {
                    warn!("Failed to search commit: {:?}", err);
                };
                vec![]
            }
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
                    .map(|doc| format!("```\n{}\n```", get_content(doc))),
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
    code: Arc<dyn CodeSearch>,
    doc: Arc<dyn DocSearch>,
    commit: Arc<dyn CommitHistorySearch>,
    context: Arc<dyn ContextService>,
    serper: Option<Box<dyn DocSearch>>,
    repository: Arc<dyn RepositoryService>,
) -> AnswerService {
    AnswerService::new(
        config, auth, chat, code, doc, commit, context, serper, repository,
    )
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
    if !config.system_prompt.is_empty() {
        output.push(ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessage {
                content: ChatCompletionRequestSystemMessageContent::Text(
                    config.system_prompt.clone(),
                ),
                name: None,
            },
        ));
    }

    for i in 0..messages.len() - 1 {
        let x = &messages[i];
        let role = match x.role {
            thread::Role::Assistant => Role::Assistant,
            thread::Role::User => Role::User,
        };

        let message: ChatCompletionRequestMessage = if role == Role::User {
            if i % 2 != 0 {
                bail!("User message must be followed by assistant message");
            }

            let y = &messages[i + 1];

            let user_attachment_input =
                user_attachment_input_from_user_message_attachment(&x.attachment);

            let content =
                build_user_prompt(&x.content, &y.attachment, Some(&user_attachment_input));
            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(
                    helper.rewrite_tag(&content),
                ),
                ..Default::default()
            })
        } else {
            ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                content: Some(ChatCompletionRequestAssistantMessageContent::Text(
                    x.content.clone(),
                )),
                ..Default::default()
            })
        };

        output.push(message);
    }

    let user_prompt = build_user_prompt(
        &messages[messages.len() - 1].content,
        attachment,
        user_attachment_input,
    );

    output.push(ChatCompletionRequestMessage::User(
        ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Text(
                helper.rewrite_tag(&user_prompt),
            ),
            ..Default::default()
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
        && assistant_attachment.code_file_list.is_none()
    {
        return user_input.to_owned();
    }

    let maybe_file_list_context = assistant_attachment
        .code_file_list
        .as_ref()
        .filter(|x| !x.file_list.is_empty())
        .map(|x| {
            format!(
                "Here is the list of files in the workspace available for reference:\n\n{}\n\n",
                x.file_list.join("\n")
            )
        })
        .unwrap_or_default();

    let maybe_file_snippet_context = {
        let snippets: Vec<String> = assistant_attachment
            .doc
            .iter()
            .map(|doc| format!("```\n{}\n```", get_content(doc)))
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

        if !citations.is_empty() {
            format!(
                "Here are the set of contexts:\n\n{}",
                citations.join("\n\n")
            )
        } else {
            String::default()
        }
    };

    format!(
        r#"You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable.

Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

Please cite the contexts with the reference numbers, in the format [[citation:x]]. If a sentence comes from multiple contexts, please list all applicable citations, like [[citation:3]][[citation:5]]. Other than code and specific names and citations, your answer must be written in the same language as the question.

{maybe_file_list_context}{maybe_file_snippet_context}

Remember, don't blindly repeat the contexts verbatim. When possible, give code snippet to demonstrate the answer. And here is the user question:

{user_input}
"#
    )
}

/// Combine code snippets from search results rather than utilizing multiple hits: Presently, there is only one rule: if the number of lines of code (LoC) is less than 300, and there are multiple hits (number of hits > 1), include the entire file.
pub async fn merge_code_snippets(
    repository: &Repository,
    hits: Vec<CodeSearchHit>,
) -> Vec<CodeSearchHit> {
    // group hits by filepath
    let mut file_hits: HashMap<String, Vec<CodeSearchHit>> = HashMap::new();
    for hit in hits.clone().into_iter() {
        let key = format!("{}-{}", repository.source_id, hit.doc.filepath);
        file_hits.entry(key).or_default().push(hit);
    }

    let mut result = Vec::with_capacity(file_hits.len());

    for (_, file_hits) in file_hits {
        // construct the full path to the file
        let path: PathBuf = repository.dir.join(&file_hits[0].doc.filepath);

        if file_hits.len() > 1 && count_lines(&path).is_ok_and(|x| x < 300) {
            let file_content = read_file_content(&path);

            if let Some(file_content) = file_content {
                debug!(
                    "The file {} is less than 300 lines, so the entire file content will be included",
                    file_hits[0].doc.filepath
                );
                let mut insert_hit = file_hits[0].clone();
                insert_hit.scores =
                    file_hits
                        .iter()
                        .fold(CodeSearchScores::default(), |mut acc, hit| {
                            acc.bm25 += hit.scores.bm25;
                            acc.embedding += hit.scores.embedding;
                            acc.rrf += hit.scores.rrf;
                            acc
                        });
                // average the scores
                let num_files = file_hits.len() as f32;
                insert_hit.scores.bm25 /= num_files;
                insert_hit.scores.embedding /= num_files;
                insert_hit.scores.rrf /= num_files;
                insert_hit.doc.body = file_content;

                // When we use entire file content, mark start_line as None.
                insert_hit.doc.start_line = None;
                result.push(insert_hit);
            }
        } else {
            result.extend(file_hits);
        }
    }

    result.sort_by(|a, b| b.scores.rrf.total_cmp(&a.scores.rrf));
    result
}

/// Read file content and return raw file content string.
pub fn read_file_content(path: &Path) -> Option<String> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(e) => {
            warn!("Error opening file {}: {}", path.display(), e);
            return None;
        }
    };
    let mut content = String::new();
    match file.read_to_string(&mut content) {
        Ok(_) => Some(content),
        Err(e) => {
            warn!("Error reading file {}: {}", path.display(), e);
            None
        }
    }
}

fn count_lines(path: &Path) -> std::io::Result<usize> {
    let mut count = 0;
    for line in BufReader::new(File::open(path)?).lines() {
        line?;
        count += 1;
    }
    Ok(count)
}

fn get_content(doc: &MessageAttachmentDoc) -> &str {
    match doc {
        MessageAttachmentDoc::Web(web) => &web.content,
        MessageAttachmentDoc::Issue(issue) => &issue.body,
        MessageAttachmentDoc::Pull(pull) => &pull.body,
    }
}

fn user_attachment_input_from_user_message_attachment(
    attachment: &MessageAttachment,
) -> MessageAttachmentInput {
    let user_attachment_code_input: Vec<MessageAttachmentCodeInput> = attachment
        .client_code
        .iter()
        .map(Clone::clone)
        .map(Into::into)
        .collect();
    MessageAttachmentInput {
        code: user_attachment_code_input,
    }
}

#[cfg(test)]
pub mod testutils;

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use juniper::ID;
    use tabby_common::{
        api::{
            code::{
                CodeSearch, CodeSearchDocument, CodeSearchHit, CodeSearchParams, CodeSearchScores,
            },
            structured_doc::{DocSearch, DocSearchDocument},
        },
        config::AnswerConfig,
    };
    use tabby_db::DbConn;
    use tabby_inference::ChatCompletionStream;
    use tabby_schema::{
        context::{ContextInfo, ContextInfoHelper, ContextService, ContextSourceValue},
        repository::{Repository, RepositoryKind},
        thread::{CodeQueryInput, CodeSearchParamsOverrideInput, DocQueryInput, MessageAttachment},
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
    use crate::service::{access_policy::testutils::make_policy, auth};

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
            code_source_id: None,
            content: content.to_owned(),
            role,
            attachment: attachment.unwrap_or_default(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }

    fn get_title(doc: &DocSearchDocument) -> &str {
        match doc {
            DocSearchDocument::Web(web_doc) => &web_doc.title,
            DocSearchDocument::Issue(issue_doc) => &issue_doc.title,
            DocSearchDocument::Pull(pull_doc) => &pull_doc.title,
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
            commit: vec![],
            client_code: vec![],
            code_file_list: None,
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
            commit: vec![],
            client_code: vec![tabby_schema::thread::MessageAttachmentClientCode {
                filepath: Some("client.py".to_owned()),
                content: "print('Hello, client!')".to_owned(),
                start_line: Some(1),
            }],
            code_file_list: Some(MessageAttachmentCodeFileList {
                file_list: vec!["client.py".to_owned(), "server.py".to_owned()],
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
        // setup minimal test repository
        let test_repo = Repository {
            id: ID::from("1".to_owned()),
            source_id: TEST_SOURCE_ID.to_owned(),
            name: "test-repo".to_string(),
            kind: RepositoryKind::Git,
            dir: PathBuf::from("test-repo"),
            git_url: TEST_GIT_URL.to_owned(),
            refs: vec![],
        };

        let context_info = ContextInfo {
            sources: vec![ContextSourceValue::Repository(test_repo)],
        };

        let test_repo = Repository {
            id: ID::from("1".to_owned()),
            source_id: TEST_SOURCE_ID.to_owned(),
            name: "test-repo".to_string(),
            kind: RepositoryKind::Git,
            dir: PathBuf::from("test-repo"),
            git_url: TEST_GIT_URL.to_owned(),
            refs: vec![],
        };

        let context_info_helper = ContextInfoHelper::new(&context_info);

        // Setup services
        let auth = Arc::new(auth::testutils::FakeAuthService::new(vec![]));
        let chat = Arc::new(FakeChatCompletionStream {
            return_error: false,
        });
        let code = Arc::new(FakeCodeSearch);
        let doc = Arc::new(FakeDocSearch);
        let context = Arc::new(FakeContextService);
        let config = make_answer_config();
        let db = DbConn::new_in_memory().await.unwrap();
        let repo_service = make_repository_service(db.clone()).await.unwrap();

        let service =
            AnswerService::new(&config, auth, chat, code, doc, context, None, repo_service);

        // Test Case 1: Basic code collection
        let input = make_code_query_input(Some(&test_repo.source_id), Some(&test_repo.git_url));
        let code_hits = service
            .collect_relevant_code(
                &test_repo,
                &context_info_helper,
                &input,
                &make_code_search_params(),
                None,
            )
            .await;
        assert!(!code_hits.is_empty(), "Should find code hits");
        assert!(code_hits[0].scores.rrf > 0.0);

        // Test Case 2: With params override
        let override_params = CodeSearchParamsOverrideInput {
            min_bm25_score: Some(0.1),
            min_embedding_score: Some(0.1),
            min_rrf_score: Some(0.1),
            num_to_return: Some(10),
            num_to_score: Some(20),
        };
        let code_hits_override = service
            .collect_relevant_code(
                &test_repo,
                &context_info_helper,
                &input,
                &make_code_search_params(),
                Some(&override_params),
            )
            .await;
        assert!(
            code_hits_override.len() >= code_hits.len(),
            "Override params should return more hits"
        );
        assert!(
            code_hits_override.iter().all(|hit| hit.scores.rrf >= 0.1),
            "All hits should meet minimum score"
        );
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

        let service = AnswerService::new(
            &config,
            auth.clone(),
            chat.clone(),
            code.clone(),
            doc.clone(),
            context.clone(),
            serper,
            repo,
        );

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

        let service = AnswerService::new(
            &config,
            auth.clone(),
            chat.clone(),
            code.clone(),
            doc.clone(),
            context.clone(),
            serper,
            repo,
        );

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
    async fn test_collect_relevant_docs() {
        let auth = Arc::new(auth::testutils::FakeAuthService::new(vec![]));
        let chat = Arc::new(FakeChatCompletionStream {
            return_error: false,
        });
        let code = Arc::new(FakeCodeSearch);
        let doc = Arc::new(FakeDocSearch);
        let context = Arc::new(FakeContextService);
        let serper = Some(Box::new(FakeDocSearch) as Box<dyn DocSearch>);
        let config = make_answer_config();
        let db = DbConn::new_in_memory().await.unwrap();
        let repo = make_repository_service(db).await.unwrap();

        let service = AnswerService::new(
            &config,
            auth.clone(),
            chat.clone(),
            code.clone(),
            doc.clone(),
            context.clone(),
            serper,
            repo,
        );

        let context_info_helper = make_context_info_helper();

        // Test Case 1: Test with valid source ID and public search enabled
        let doc_query_1 = DocQueryInput {
            content: "Test query[[source:source-1]]".to_string(),
            source_ids: Some(vec!["source-1".to_string()]),
            search_public: true,
        };

        let hits_1 = service
            .collect_relevant_docs(&context_info_helper, &doc_query_1)
            .await;

        assert_eq!(hits_1.len(), 10);
        assert!(hits_1.iter().any(|hit| get_title(&hit.doc) == "Document 1"));

        // Test Case 2: Test with invalid source ID
        let doc_query_2 = DocQueryInput {
            content: "Test query".to_string(),
            source_ids: Some(vec!["invalid-source".to_string()]),
            search_public: false,
        };

        let hits_2 = service
            .collect_relevant_docs(&context_info_helper, &doc_query_2)
            .await;

        assert_eq!(hits_2.len(), 0);

        // Test Case 3: Test with no source IDs but public search
        let doc_query_3 = DocQueryInput {
            content: "Test query".to_string(),
            source_ids: None,
            search_public: true,
        };

        let hits_3 = service
            .collect_relevant_docs(&context_info_helper, &doc_query_3)
            .await;

        assert!(!hits_3.is_empty());

        // Test Case 4: Test with empty source IDs and no public search
        let doc_query_4 = DocQueryInput {
            content: "Test query".to_string(),
            source_ids: Some(vec![]),
            search_public: false,
        };

        let hits_4 = service
            .collect_relevant_docs(&context_info_helper, &doc_query_4)
            .await;

        assert_eq!(hits_4.len(), 0);
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
        let service = Arc::new(AnswerService::new(
            &config, auth, chat, code, doc, context, serper, repo,
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

        let service = AnswerService::new(
            &config,
            auth,
            chat,
            code,
            doc,
            context,
            serper,
            repo_service,
        );

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

    #[tokio::test]
    async fn test_merge_code_snippets() {
        let db = DbConn::new_in_memory().await.unwrap();
        let repo_service = make_repository_service(db.clone()).await.unwrap();

        let git_url = "https://github.com/test/repo.git".to_string();
        let _id = repo_service
            .git()
            .create("repo".to_string(), git_url.clone())
            .await
            .unwrap();

        let policy = make_policy(db.clone()).await;
        let repo = repo_service
            .repository_list(Some(&policy))
            .await
            .unwrap()
            .pop();

        let hits = vec![
            CodeSearchHit {
                doc: CodeSearchDocument {
                    file_id: "file1".to_string(),
                    chunk_id: "chunk1".to_string(),
                    body: "fn test1() {}\nfn test2() {}".to_string(),
                    filepath: "test.rs".to_string(),
                    git_url: "https://github.com/test/repo.git".to_string(),
                    commit: Some("commit".to_string()),
                    language: "rust".to_string(),
                    start_line: Some(1),
                },
                scores: CodeSearchScores {
                    bm25: 0.5,
                    embedding: 0.7,
                    rrf: 0.3,
                },
            },
            CodeSearchHit {
                doc: CodeSearchDocument {
                    file_id: "file1".to_string(),
                    chunk_id: "chunk2".to_string(),
                    body: "fn test3() {}\nfn test4() {}".to_string(),
                    filepath: "test.rs".to_string(),
                    git_url: "https://github.com/test/repo.git".to_string(),
                    commit: Some("commit".to_string()),
                    language: "rust".to_string(),
                    start_line: Some(3),
                },
                scores: CodeSearchScores {
                    bm25: 0.6,
                    embedding: 0.8,
                    rrf: 0.4,
                },
            },
        ];

        let result = merge_code_snippets(&repo.unwrap(), hits).await;

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].doc.commit, Some("commit".to_string()));
        assert_eq!(result[1].doc.commit, Some("commit".to_string()));
    }
}
