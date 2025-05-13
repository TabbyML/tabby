mod prompt_tools;

use std::sync::Arc;

use anyhow::anyhow;
use async_openai_alt::types::ChatCompletionRequestMessage;
use async_stream::stream;
use async_trait::async_trait;
use chrono::Utc;
use futures::{stream::BoxStream, StreamExt};
use juniper::ID;
use prompt_tools::{
    prompt_page_content, prompt_page_section_content, prompt_page_section_titles, prompt_page_title,
};
use tabby_common::{api::structured_doc::DocSearchDocument, config::PageConfig};
use tabby_db::{DbConn, PageSectionDAO};
use tabby_inference::ChatCompletionStream;
use tabby_schema::{
    auth::AuthenticationService,
    context::ContextService,
    page::{
        AttachmentCodeQueryDebugData, AttachmentDocQueryDebugData, CreatePageRunInput,
        CreatePageSectionRunInput, CreateThreadToPageRunInput, MoveSectionDirection, Page,
        PageCompleted, PageContentCompleted, PageContentDebugData, PageContentDelta, PageCreated,
        PageRunDebugOptionInput, PageRunItem, PageRunStream, PageSection,
        PageSectionAttachmentCode, PageSectionAttachmentCodeFileList, PageSectionAttachmentDoc,
        PageSectionContentCompleted, PageSectionContentDebugData, PageSectionContentDelta,
        PageSectionDebugData, PageSectionRunDebugOptionInput, PageSectionsCreated, PageService,
        PageTitleDebugData, SectionAttachment, SectionRunItem, SectionRunStream,
        ThreadToPageRunStream,
    },
    policy::AccessPolicy,
    retrieval::{AttachmentCodeFileList, AttachmentDocHit},
    thread::{
        CodeQueryInput, DocQueryInput, Message, MessageAttachment, MessageAttachmentCodeFileList,
        MessageAttachmentDoc, Role, ThreadRunItem, ThreadRunOptionsInput, ThreadService,
    },
    AsID, AsRowid, ChatCompletionMessage, CoreError, Result,
};
use tracing::error;

use super::{
    retrieval::{attachment_doc_from_search, attachment_docs_from_db},
    utils::prompt::{request_llm_stream, transform_line_items},
};
use crate::service::{
    answer::AnswerService,
    graphql_pagination_to_filter,
    retrieval::RetrievalService,
    utils::{
        convert_messages_to_chat_completion_request,
        convert_user_message_to_chat_completion_request, get_source_id,
        prompt::{request_llm_with_message, trim_title},
    },
};

struct PageServiceImpl {
    config: PageConfig,
    db: DbConn,
    auth: Arc<dyn AuthenticationService>,
    chat: Arc<dyn ChatCompletionStream>,
    thread: Arc<dyn ThreadService>,
    context: Arc<dyn ContextService>,
    retrieval: Arc<RetrievalService>,
    answer: Arc<AnswerService>,
}

pub fn create(
    config: PageConfig,
    db: DbConn,
    auth: Arc<dyn AuthenticationService>,
    chat: Arc<dyn ChatCompletionStream>,
    thread: Arc<dyn ThreadService>,
    context: Arc<dyn ContextService>,
    retrieval: Arc<RetrievalService>,
    answer: Arc<AnswerService>,
) -> impl PageService {
    PageServiceImpl {
        config,
        db,
        auth,
        chat,
        thread,
        context,
        retrieval,
        answer,
    }
}

#[async_trait]
impl PageService for PageServiceImpl {
    async fn convert_thread_to_page(
        &self,
        policy: &AccessPolicy,
        author_id: &ID,
        input: &CreateThreadToPageRunInput,
    ) -> Result<ThreadToPageRunStream> {
        let _thread = self
            .thread
            .get(&input.thread_id)
            .await?
            .ok_or_else(|| CoreError::NotFound("Thread not found"))?;

        let thread_messages = self
            .thread
            .list_thread_messages(&input.thread_id, None, None, None, None)
            .await?;

        // use the first message source id for page
        let code_source_id = thread_messages
            .iter()
            .find_map(|m| m.code_source_id.as_deref())
            .map(ToOwned::to_owned);

        let debug_option = input
            .debug_option
            .as_ref()
            .map(|d| PageRunDebugOptionInput {
                return_chat_completion_request: d.return_chat_completion_request,
                return_query_request: d.return_query_request,
            });

        self.page_run(
            policy,
            author_id,
            None,
            code_source_id.as_deref(),
            None,
            Some(&thread_messages),
            debug_option.as_ref(),
        )
        .await
    }

    async fn create_run(
        &self,
        policy: &AccessPolicy,
        author_id: &ID,
        input: &CreatePageRunInput,
    ) -> Result<PageRunStream> {
        let user = self.auth.get_user(author_id).await?;

        let now = Utc::now();
        let mut answer_messages = vec![Message {
            id: ID::new("page-run-user-msg"),
            thread_id: ID::new("page-run-temporary-thread"),
            code_source_id: None,
            content: input.title_prompt.clone(),
            role: Role::User,
            attachment: MessageAttachment::default(),
            created_at: now,
            updated_at: now,
        }];

        let answer_options = ThreadRunOptionsInput {
            model_name: None,
            code_query: input.code_query.clone(),
            doc_query: input.doc_query.clone(),
            generate_relevant_questions: false,
            debug_options: None,
        };

        let answer = self.answer.clone();
        let mut answer_stream = answer
            .answer(&user, &answer_messages, &answer_options, None)
            .await?;

        let mut assistant_content = String::new();
        let mut assistant_attachment = MessageAttachment::default();

        while let Some(item_result) = answer_stream.next().await {
            match item_result {
                Ok(item) => match item {
                    ThreadRunItem::ThreadAssistantMessageContentDelta(delta) => {
                        assistant_content.push_str(&delta.delta);
                    }
                    ThreadRunItem::ThreadAssistantMessageAttachmentsCode(code_attachments) => {
                        assistant_attachment.code = code_attachments
                            .hits
                            .into_iter()
                            .map(|hit| hit.code)
                            .collect();
                    }
                    ThreadRunItem::ThreadAssistantMessageAttachmentsDoc(doc_attachments) => {
                        assistant_attachment.doc = doc_attachments
                            .hits
                            .into_iter()
                            .map(|hit| hit.doc)
                            .collect();
                    }
                    ThreadRunItem::ThreadAssistantMessageAttachmentsCodeFileList(
                        file_list_attachment,
                    ) => {
                        assistant_attachment.code_file_list = Some(MessageAttachmentCodeFileList {
                            file_list: file_list_attachment.file_list,
                            truncated: file_list_attachment.truncated,
                        });
                    }
                    // skip all other items
                    _ => {}
                },
                Err(e) => {
                    error!("Error from AnswerService stream: {:?}", e);
                    return Err(e);
                }
            }
        }

        let now = Utc::now();
        let assistant_message = Message {
            id: ID::new("page-run-assistant-msg"),
            thread_id: ID::new("page-run-temporary-thread"),
            code_source_id: None,
            content: assistant_content,
            role: Role::Assistant,
            attachment: assistant_attachment,
            created_at: now,
            updated_at: now,
        };

        answer_messages.push(assistant_message);

        let code_source_id = if let Some(code_query) = input.code_query.as_ref() {
            get_source_id(self.context.clone(), policy, code_query).await
        } else {
            None
        };

        self.page_run(
            policy,
            author_id,
            Some(&input.title_prompt),
            code_source_id.as_deref(),
            input.doc_query.as_ref(),
            Some(&answer_messages),
            input.debug_option.as_ref(),
        )
        .await
    }

    async fn append_section(
        &self,
        policy: &AccessPolicy,
        input: &CreatePageSectionRunInput,
    ) -> Result<SectionRunStream> {
        let existing_page_sections = self
            .list_sections(&input.page_id, None, None, None, None)
            .await?;

        let page_id = input.page_id.as_rowid()?;
        let new_section_prompt = input.title_prompt.clone();

        let page = self
            .db
            .get_page(page_id)
            .await?
            .ok_or_else(|| CoreError::NotFound("Page not found"))?;
        let page_title = page.title.unwrap_or_default();
        let code_source_id = page.code_source_id.clone();
        let doc_query = input.doc_query.clone();

        let (new_section_title, section_title_debug_messages) = generate_page_sections(
            input
                .debug_option
                .as_ref()
                .map(|x| x.return_chat_completion_request)
                .unwrap_or(false),
            1,
            self.chat.clone(),
            self.context.clone(),
            policy,
            Some(&new_section_prompt),
            &page_title,
            &existing_page_sections,
            None,
        )
        .await?;
        let new_section_title = new_section_title
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Failed to generate section title"))?;

        let section = self
            .db
            .create_page_section(page_id, &new_section_title)
            .await?;
        let section_id = section.id;

        let db = self.db.clone();
        let policy = policy.clone();
        let chat = self.chat.clone();
        let context = self.context.clone();
        let retrieval = self.retrieval.clone();
        let config = self.config.clone();
        let auth = self.auth.clone();
        let debug_option = input.debug_option.clone();

        let s = stream! {
            // Create a channel that will be closed when the client disconnects
            let (tx, rx) = tokio::sync::oneshot::channel::<()>();
            let _cleanup_trigger = tx;
            let db_cleanup = db.clone();
            let _disconnect_guard = tokio::spawn(async move {
                rx.await.unwrap_or_default();
                let _ = db_cleanup.delete_page_sections_without_content(page_id).await;
            });

            let mut section_from_db = section_from_db(auth.clone(), section).await;
            if debug_option.as_ref().map(|x| x.return_chat_completion_request).unwrap_or(false) {
                section_from_db.debug_data = section_title_debug_messages.map(|messages| PageSectionDebugData {
                    generate_section_titles_messages: messages,
                });
            }
            yield Ok(SectionRunItem::PageSectionCreated(section_from_db.into()));

            let mut attachments_stream = generate_section_with_attachments(
                debug_option.as_ref(),
                policy,
                &page_id.as_id(),
                section_id,
                new_section_title,
                Some(new_section_prompt),
                &existing_page_sections,
                &page_title,
                code_source_id.as_deref(),
                doc_query.as_ref(),
                None,
                db,
                auth,
                chat,
                context,
                retrieval,
                config,
            )
            .await?;


            while let Some(item) = attachments_stream.as_mut().next().await {
                yield item;
            }
        };

        Ok(s.boxed())
    }

    async fn delete(&self, id: &ID) -> Result<()> {
        self.db.delete_page(id.as_rowid()?).await?;
        Ok(())
    }

    async fn get(&self, id: &ID) -> Result<Page> {
        let page = self
            .db
            .get_page(id.as_rowid()?)
            .await?
            .ok_or_else(|| CoreError::NotFound("Page not found"))?;
        Ok(page.into())
    }

    async fn update_title(&self, id: &ID, title: &str) -> Result<()> {
        self.db.update_page_title(id.as_rowid()?, title).await?;
        Ok(())
    }

    async fn update_content(&self, id: &ID, content: &str) -> Result<()> {
        self.db.update_page_content(id.as_rowid()?, content).await?;
        Ok(())
    }

    async fn list(
        &self,
        ids: Option<&[ID]>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<Page>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;

        let ids = ids.map(|x| {
            x.iter()
                .filter_map(|x| x.as_rowid().ok())
                .collect::<Vec<_>>()
        });
        let pages = self
            .db
            .list_pages(ids.as_deref(), limit, skip_id, backwards)
            .await?;

        Ok(pages.into_iter().map(Into::into).collect())
    }

    async fn list_sections(
        &self,
        page_id: &ID,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<PageSection>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;

        let sections = self
            .db
            .list_page_sections(page_id.as_rowid()?, limit, skip_id, backwards)
            .await?;

        let mut result = Vec::with_capacity(sections.len());
        for section in sections {
            result.push(section_from_db(self.auth.clone(), section).await);
        }
        Ok(result)
    }

    async fn get_section(&self, id: &ID) -> Result<PageSection> {
        let section = self
            .db
            .get_page_section(id.as_rowid()?)
            .await?
            .ok_or_else(|| CoreError::NotFound("Section not found"))?;
        Ok(section_from_db(self.auth.clone(), section).await)
    }

    async fn delete_section(&self, id: &ID) -> Result<()> {
        self.db.delete_page_section(id.as_rowid()?).await?;
        Ok(())
    }

    async fn move_section(
        &self,
        page_id: &ID,
        id: &ID,
        direction: MoveSectionDirection,
    ) -> Result<()> {
        self.db
            .move_page_section(
                page_id.as_rowid()?,
                id.as_rowid()?,
                matches!(direction, MoveSectionDirection::Up),
            )
            .await?;
        Ok(())
    }

    async fn update_section_title(&self, id: &ID, title: &str) -> Result<()> {
        self.db
            .update_page_section_title(id.as_rowid()?, title)
            .await?;
        Ok(())
    }

    async fn update_section_content(&self, id: &ID, content: &str) -> Result<()> {
        self.db
            .update_page_section_content(id.as_rowid()?, content)
            .await?;
        Ok(())
    }
}

impl PageServiceImpl {
    async fn page_run(
        &self,
        policy: &AccessPolicy,
        author_id: &ID,
        title_prompt: Option<&str>,
        code_source_id: Option<&str>,
        doc_query: Option<&DocQueryInput>,
        thread_messages: Option<&[Message]>,
        debug_option: Option<&PageRunDebugOptionInput>,
    ) -> Result<PageRunStream> {
        let page_id = self
            .db
            .create_page(author_id.as_rowid()?, code_source_id)
            .await?
            .as_id();

        let thread_messages = thread_messages.map(|messages| {
            filter_out_messages_doc_with_self_id(page_id.to_string().as_ref(), messages)
        });

        let debug_return_chat_request = debug_option
            .map(|x| x.return_chat_completion_request)
            .unwrap_or_else(|| false);

        let (page_title, title_debug_messages) = self
            .generate_page_title(
                debug_return_chat_request,
                policy,
                &page_id,
                title_prompt,
                thread_messages.as_deref(),
            )
            .await?;

        let code_source_id = code_source_id.map(ToOwned::to_owned);
        let doc_query = if let Some(doc_query) = doc_query {
            Some(doc_query.to_owned())
        } else {
            code_source_id.as_ref().map(|code_source_id| DocQueryInput {
                content: title_prompt.unwrap_or_else(|| &page_title).to_owned(),
                search_public: false,
                source_ids: Some(vec![code_source_id.clone()]),
            })
        };

        let db = self.db.clone();
        let policy = policy.clone();
        let chat = self.chat.clone();
        let context = self.context.clone();
        let author_id = author_id.clone();
        let retrieval = self.retrieval.clone();
        let config = self.config.clone();
        let auth = self.auth.clone();
        let page_rowid = page_id.as_rowid()?;
        let debug_option = debug_option.map(|x| PageSectionRunDebugOptionInput {
            return_chat_completion_request: x.return_chat_completion_request,
            return_query_request: x.return_query_request,
        });

        let s = stream! {
            // Create a channel that will be closed when the client disconnects
            let (tx, rx) = tokio::sync::oneshot::channel::<()>();
            let _cleanup_trigger = tx;
            let db_cleanup = db.clone();
            let _disconnect_guard = tokio::spawn(async move {
                rx.await.unwrap_or_default();
                let _ = db_cleanup.delete_page_sections_without_content(page_rowid).await;
            });

            yield Ok(PageRunItem::PageCreated(PageCreated {
                id: page_id.clone(),
                author_id: author_id.clone(),
                title: page_title.clone(),
                debug_data: title_debug_messages.map(|messages| PageTitleDebugData {
                    generate_page_title_messages: messages,
                }),
            }));

            let (page_section_titles, section_titles_debug_messages) = generate_page_sections(
                debug_return_chat_request,
                3,
                chat.clone(),
                context.clone(),
                &policy,
                None,
                &page_title,
                &[],
                thread_messages.as_deref(),
            ).await?;
            let mut page_sections = Vec::new();
            for section_title in &page_section_titles {
                let section = db.create_page_section(page_id.as_rowid()?, section_title).await?;
                page_sections.push(section_from_db(auth.clone(), section).await.into());
            }

            yield Ok(PageRunItem::PageSectionsCreated(PageSectionsCreated {
                sections: page_sections.clone(),

                debug_data: section_titles_debug_messages.map(|messages| PageSectionDebugData {
                    generate_section_titles_messages: messages,
                }),
            }));

            let (content_stream, content_debug_messages) = generate_page_content(
                debug_return_chat_request,
                chat.clone(),
                context.clone(),
                &policy,
                &page_title,
                &page_section_titles,
                thread_messages.as_deref()
            ).await?;
            for await delta in content_stream {
                let delta = delta?;
                db.append_page_content(page_id.as_rowid()?, &delta).await?;
                yield Ok(PageRunItem::PageContentDelta(PageContentDelta{
                    delta
                }));
            }

            yield Ok(PageRunItem::PageContentCompleted(PageContentCompleted {
                id: page_id.clone(),

                debug_data: content_debug_messages.map(|messages| PageContentDebugData {
                    generate_page_content_messages: messages,
                }),
            }));

            for (i, section) in page_sections.iter().enumerate() {
                let sections_in_db: Vec<_> = db
                    .list_page_sections(page_id.as_rowid()?, None, None, false)
                    .await?;
                let existed_sections = sections_from_db(auth.clone(), sections_in_db).await.into_iter().take(i).collect::<Vec<_>>();

                let attachments_stream = generate_section_with_attachments(
                    debug_option.as_ref(),
                    policy.clone(),
                    &page_id,
                    section.id.as_rowid()?,
                    section.title.clone(),
                    None,
                    &existed_sections,
                    &page_title,
                    code_source_id.as_deref(),
                    doc_query.as_ref(),
                    thread_messages.as_deref(),
                    db.clone(),
                    auth.clone(),
                    chat.clone(),
                    context.clone(),
                    retrieval.clone(),
                    config.clone(),
                ).await?;
                let mut section_stream = Box::pin(attachments_stream.map(|item| {
                    item.map(|section_item| match section_item {
                        SectionRunItem::PageSectionCreated(_) => None,
                        SectionRunItem::PageSectionAttachmentCodeFileList(data) =>
                            Some(PageRunItem::PageSectionAttachmentCodeFileList(data)),
                        SectionRunItem::PageSectionAttachmentCode(data) =>
                            Some(PageRunItem::PageSectionAttachmentCode(data)),
                        SectionRunItem::PageSectionContentDelta(data) =>
                            Some(PageRunItem::PageSectionContentDelta(data)),
                        SectionRunItem::PageSectionContentCompleted(data) =>
                            Some(PageRunItem::PageSectionContentCompleted(data)),
                        SectionRunItem::PageSectionAttachmentDoc(data) =>
                            Some(PageRunItem::PageSectionAttachmentDoc(data)),
                    })
                }).filter_map(|item| async move {
                    match item {
                        Ok(Some(run_item)) => Some(Ok(run_item)),
                        Ok(None) => None,
                        Err(e) => Some(Err(e)),
                    }
                }));

                while let Some(item) = section_stream.as_mut().next().await {
                    yield item;
                }
            }

            yield Ok(PageRunItem::PageCompleted(PageCompleted {
                id: page_id,
            }));
        };

        Ok(s.boxed())
    }

    async fn generate_page_title(
        &self,
        debug: bool,
        policy: &AccessPolicy,
        page_id: &ID,
        title_prompt: Option<&str>,
        thread_messages: Option<&[Message]>,
    ) -> Result<(String, Option<Vec<ChatCompletionMessage>>)> {
        let messages = build_chat_messages(
            self.context.clone(),
            policy,
            thread_messages,
            prompt_page_title(title_prompt).as_str(),
            &SectionAttachment::default(),
        )
        .await?;

        let debug_messages: Option<Vec<ChatCompletionMessage>> = if debug {
            Some(messages.clone().into_iter().map(Into::into).collect())
        } else {
            None
        };

        let title = request_llm_with_message(self.chat.clone(), messages).await?;
        let title = trim_title(title.as_str());

        self.db
            .update_page_title(page_id.as_rowid()?, title)
            .await?;
        Ok((title.to_owned(), debug_messages))
    }
}

async fn sections_from_db(
    auth: Arc<dyn AuthenticationService>,
    sections: Vec<PageSectionDAO>,
) -> Vec<PageSection> {
    let mut result = Vec::with_capacity(sections.len());
    for section in sections {
        result.push(section_from_db(auth.clone(), section).await);
    }
    result
}

async fn section_from_db(
    auth: Arc<dyn AuthenticationService>,
    value: PageSectionDAO,
) -> PageSection {
    let (code, code_file_list, doc) = if let Some(attachment) = value.attachment {
        let code = attachment
            .0
            .code
            .map(|x| x.iter().map(|x| x.into()).collect());
        let code_file_list = attachment.0.code_file_list.map(|x| x.into());
        let doc = if let Some(doc) = attachment.0.doc {
            Some(attachment_docs_from_db(auth, doc).await)
        } else {
            None
        };

        (code, code_file_list, doc)
    } else {
        (None, None, None)
    };

    PageSection {
        id: value.id.as_id(),
        page_id: value.page_id.as_id(),
        title: value.title,
        position: value.position as i32,
        content: value.content.unwrap_or_default(),
        created_at: value.created_at,
        updated_at: value.updated_at,

        attachments: SectionAttachment {
            code: code.unwrap_or_default(),
            code_file_list,
            doc: doc.unwrap_or_default(),
        },

        debug_data: None,
    }
}

async fn build_chat_messages(
    context: Arc<dyn ContextService>,
    policy: &AccessPolicy,
    thread_messages: Option<&[Message]>,
    user_message: &str,
    user_message_attachment: &SectionAttachment,
) -> Result<Vec<ChatCompletionRequestMessage>> {
    let helper = context.read(Some(policy)).await?.helper();
    let mut messages = Vec::new();
    if let Some(thread_messages) = thread_messages {
        messages.extend(
            convert_messages_to_chat_completion_request(None, &helper, thread_messages)?
                .into_iter(),
        );
    }

    messages.push(convert_user_message_to_chat_completion_request(
        &helper,
        user_message,
        &user_message_attachment.clone().into(),
        None,
    ));

    Ok(messages)
}

fn filter_out_messages_doc_with_self_id(page_id: &str, messages: &[Message]) -> Vec<Message> {
    messages
        .iter()
        .map(|message| {
            let mut message_clone = message.clone();
            // Filter out any attachment of type "Page" with id equal to page_id
            message_clone.attachment.doc = message_clone
                .attachment
                .doc
                .iter()
                .filter(|doc| {
                    // Keep docs that are not of type "Page" or have different id
                    if let MessageAttachmentDoc::Page(doc) = doc {
                        doc.link == page_id
                    } else {
                        false
                    }
                })
                .cloned()
                .collect();

            message_clone
        })
        .collect()
}

async fn generate_page_content(
    debug: bool,
    chat: Arc<dyn ChatCompletionStream>,
    context: Arc<dyn ContextService>,
    policy: &AccessPolicy,
    page_title: &str,
    page_section_titles: &[String],
    thread_messages: Option<&[Message]>,
) -> Result<(
    BoxStream<'static, tabby_schema::Result<String>>,
    Option<Vec<ChatCompletionMessage>>,
)> {
    let messages = build_chat_messages(
        context,
        policy,
        thread_messages,
        prompt_page_content(page_title, page_section_titles).as_str(),
        &SectionAttachment::default(),
    )
    .await?;

    let debug_messages: Option<Vec<ChatCompletionMessage>> = if debug {
        Some(messages.clone().into_iter().map(Into::into).collect())
    } else {
        None
    };

    Ok((
        request_llm_stream(chat.clone(), messages).await,
        debug_messages,
    ))
}

async fn generate_section_with_attachments(
    debug: Option<&PageSectionRunDebugOptionInput>,
    policy: AccessPolicy,
    page_id: &ID,
    section_id: i64,
    section_title: String,
    section_prompt: Option<String>,
    existing_sections: &[PageSection],
    page_title: &str,
    code_source_id: Option<&str>,
    doc_query: Option<&DocQueryInput>,
    thread_messages: Option<&[Message]>,
    db: DbConn,
    auth: Arc<dyn AuthenticationService>,
    chat: Arc<dyn ChatCompletionStream>,
    context: Arc<dyn ContextService>,
    retrieval: Arc<RetrievalService>,
    config: PageConfig,
) -> Result<BoxStream<'static, Result<SectionRunItem>>> {
    let policy = policy.clone();
    let existing_sections = existing_sections.to_vec();
    let page_id = page_id.to_string();
    let page_title = page_title.to_string();
    let thread_messages = thread_messages.map(ToOwned::to_owned);
    let context_info_helper = context.read(Some(&policy)).await?.helper();
    let code_source_id = code_source_id.map(ToOwned::to_owned);
    let doc_query = doc_query.map(ToOwned::to_owned);
    let debug_chat_request = debug
        .map(|d| d.return_chat_completion_request)
        .unwrap_or(false);
    let debug_query_request = debug.map(|d| d.return_query_request).unwrap_or(false);

    let stream = stream! {
        let mut attachment = SectionAttachment::default();
        if let Some(source_id) = &code_source_id {
            // File list attachment logic
            if false {
                match retrieval.collect_file_list_by_source_id(&policy, source_id, None, Some(300)).await {
                    Ok((file_list, truncated)) => {
                        attachment.code_file_list = Some(AttachmentCodeFileList {
                            file_list: file_list.clone(),
                            truncated,
                        });
                        db.update_page_section_code_file_list(section_id, &file_list, truncated).await?;
                        yield Ok(SectionRunItem::PageSectionAttachmentCodeFileList(PageSectionAttachmentCodeFileList {
                            id: section_id.as_id(),
                            code_file_list: AttachmentCodeFileList{
                                file_list,
                                truncated
                            }
                        }));
                    }
                    Err(e) => {
                        error!("failed to list files for repository {}: {}", source_id, e);
                    }
                }
            }

            // Code attachment logic
            let query_content = section_prompt.unwrap_or_else(|| section_title.clone());
            let hits = retrieval.collect_relevant_code_by_source_id(
                &policy,
                source_id,
                &context_info_helper,
                &CodeQueryInput {
                    source_id: Some(source_id.clone()),
                    content: query_content.clone(),
                    ..Default::default()
                },
                &config.code_search_params,
                None,
            ).await?;
            let debug_data = if debug_query_request {
                Some(AttachmentCodeQueryDebugData {
                    source_id: source_id.clone(),
                    query: query_content,
                })
            } else {
                None
            };

            attachment.code = hits.iter().map(|x| x.doc.clone().into()).collect::<Vec<_>>();

            let hits = hits.into_iter().map(|x| x.into()).collect::<Vec<_>>();
            if !hits.is_empty() {
                db.update_page_section_code_attachments(section_id, &attachment.code.iter().map(|c| c.into()).collect::<Vec<_>>()).await?;
            }
            yield Ok(SectionRunItem::PageSectionAttachmentCode(
                PageSectionAttachmentCode {
                    id: section_id.as_id(),
                    codes: hits,

                    debug_data,
                }
            ));
        }

        if let Some(doc_query) = doc_query {
            let hits = retrieval.collect_relevant_docs(
                &context_info_helper,
                &doc_query,
            ).await;
            let hits = hits.iter().filter(|x| {
                if let DocSearchDocument::Page(doc) = &x.doc {
                    doc.link != page_id
                } else {
                    true
                }
            }).collect::<Vec<_>>();
            let debug_data = if debug_query_request {
                Some(AttachmentDocQueryDebugData {
                    source_ids: doc_query.source_ids.clone().unwrap_or_default(),
                    query: doc_query.content.clone(),
                })
            } else {
                None
            };

            let hits = futures::future::join_all(hits.into_iter().map(|x| {
                let score = x.score;
                let doc = x.doc.clone();
                let auth = auth.clone();
                async move {
                    AttachmentDocHit {
                        score: score as f64,
                        doc: attachment_doc_from_search(auth, doc).await,
                    }
                }
            })).await;
            if !hits.is_empty() {
                attachment.doc = hits.clone().into_iter().map(|x| x.doc).collect::<Vec<_>>();
                db.update_page_section_doc_attachments(section_id, &attachment.doc.iter().map(|c| c.into()).collect::<Vec<_>>()).await?;
            }

            yield Ok(SectionRunItem::PageSectionAttachmentDoc(
                PageSectionAttachmentDoc {
                    id: section_id.as_id(),
                    doc: hits,

                    debug_data,
                }
            ));
        }

        // Generate section content
        let (content_stream, debug_messages) = generate_page_section_content(
            debug_chat_request,
            chat.clone(),
            context.clone(),
            &policy,
            thread_messages.as_deref(),
            &page_title,
            &existing_sections,
            &section_title,
            &attachment,
        ).await?;

        for await delta in content_stream {
            let delta = delta?;
            db.append_page_section_content(section_id, &delta).await?;
            yield Ok(SectionRunItem::PageSectionContentDelta(PageSectionContentDelta{
                id: section_id.as_id(),
                delta
            }));
        }

        yield Ok(SectionRunItem::PageSectionContentCompleted(PageSectionContentCompleted {
            id: section_id.as_id(),

            debug_data: debug_messages.map(|messages| PageSectionContentDebugData {
                generate_section_content_messages: messages,
            }),
        }));
    };

    Ok(stream.boxed())
}

pub async fn generate_page_sections(
    debug: bool,
    count: usize,
    chat: Arc<dyn ChatCompletionStream>,
    context: Arc<dyn ContextService>,
    policy: &AccessPolicy,
    new_section_prompt: Option<&str>,
    page_title: &str,
    page_sections: &[PageSection],
    thread_messages: Option<&[Message]>,
) -> anyhow::Result<(Vec<String>, Option<Vec<ChatCompletionMessage>>)> {
    let messages = build_chat_messages(
        context,
        policy,
        thread_messages,
        prompt_page_section_titles(count, page_title, page_sections, new_section_prompt).as_str(),
        &SectionAttachment::default(),
    )
    .await?;

    let debug_messages: Option<Vec<ChatCompletionMessage>> = if debug {
        Some(messages.clone().into_iter().map(Into::into).collect())
    } else {
        None
    };

    let titles = request_llm_with_message(chat.clone(), messages).await?;
    Ok((
        transform_line_items(&titles)
            .into_iter()
            .map(|x| trim_title(x.as_str()).to_owned())
            .collect(),
        debug_messages,
    ))
}

pub async fn generate_page_section_content(
    debug: bool,
    chat: Arc<dyn ChatCompletionStream>,
    context: Arc<dyn ContextService>,
    policy: &AccessPolicy,
    thread_messages: Option<&[Message]>,
    page_title: &str,
    page_sections: &[PageSection],
    new_section_title: &str,
    new_section_attachment: &SectionAttachment,
) -> Result<(
    BoxStream<'static, tabby_schema::Result<String>>,
    Option<Vec<ChatCompletionMessage>>,
)> {
    let messages = build_chat_messages(
        context,
        policy,
        thread_messages,
        prompt_page_section_content(page_title, page_sections, new_section_title).as_str(),
        new_section_attachment,
    )
    .await?;

    let debug_messages: Option<Vec<ChatCompletionMessage>> = if debug {
        Some(messages.clone().into_iter().map(Into::into).collect())
    } else {
        None
    };

    Ok((
        request_llm_stream(chat.clone(), messages).await,
        debug_messages,
    ))
}

#[cfg(test)]
mod tests {
    use crate::{answer, FakeAuthService};

    #[tokio::test]
    async fn test_move_section() {
        use tabby_common::api::structured_doc::DocSearch;
        use tabby_db::DbConn;
        use tabby_schema::page::MoveSectionDirection;

        use super::*;
        use crate::{
            answer::testutils::{
                make_answer_config, make_repository_service, FakeChatCompletionStream,
                FakeCodeSearch, FakeContextService, FakeDocSearch,
            },
            event_logger::test_utils::MockEventLogger,
            retrieval,
            service::{setting, thread},
        };

        let db = DbConn::new_in_memory().await.unwrap();
        let user_id = db
            .create_user("test@example.com".into(), None, true, None)
            .await
            .unwrap();
        let page_id = db.create_page(user_id, None).await.unwrap();
        let section0 = db.create_page_section(page_id, "Section 0").await.unwrap();
        assert_eq!(section0.position, 0);
        let section1 = db.create_page_section(page_id, "Section 1").await.unwrap();
        assert_eq!(section1.position, 1);
        let section2 = db.create_page_section(page_id, "Section 2").await.unwrap();
        assert_eq!(section2.position, 2);

        let chat: Arc<dyn ChatCompletionStream> = Arc::new(FakeChatCompletionStream {
            return_error: false,
        });
        let context: Arc<dyn ContextService> = Arc::new(FakeContextService);
        let thread = Arc::new(thread::create(db.clone(), None, None, context.clone()));
        let context: Arc<dyn ContextService> = Arc::new(FakeContextService);
        let code = Arc::new(FakeCodeSearch);
        let doc = Arc::new(FakeDocSearch);
        let serper = Some(Box::new(FakeDocSearch) as Box<dyn DocSearch>);
        let repo_service = make_repository_service(db.clone()).await.unwrap();
        let auth = Arc::new(FakeAuthService::new(vec![]));
        let settings = Arc::new(setting::create(db.clone()));

        let retrieval = Arc::new(retrieval::create(
            code.clone(),
            doc.clone(),
            serper,
            repo_service.clone(),
            settings,
        ));

        let logger = Arc::new(MockEventLogger);
        let answer_config = make_answer_config();
        let answer = Arc::new(answer::create(
            logger,
            &answer_config,
            auth.clone(),
            chat.clone(),
            retrieval.clone(),
            context.clone(),
        ));
        let service = create(
            PageConfig::default(),
            db,
            auth,
            chat,
            thread,
            context,
            retrieval,
            answer,
        );

        // move down
        service
            .move_section(
                &page_id.as_id(),
                &section0.id.as_id(),
                MoveSectionDirection::Down,
            )
            .await
            .unwrap();
        let sections = service
            .list_sections(&page_id.as_id(), None, None, None, None)
            .await
            .unwrap();
        assert_eq!(0, sections[1].position as i64);
        assert_eq!(1, sections[0].position as i64);
        assert_eq!(2, sections[2].position as i64);

        // move up
        service
            .move_section(
                &page_id.as_id(),
                &section2.id.as_id(),
                MoveSectionDirection::Up,
            )
            .await
            .unwrap();
        let sections = service
            .list_sections(&page_id.as_id(), None, None, None, None)
            .await
            .unwrap();
        assert_eq!(0, sections[1].position as i64);
        assert_eq!(1, sections[2].position as i64);
        assert_eq!(2, sections[0].position as i64);

        // move the first section up, should return error
        assert!(service
            .move_section(
                &page_id.as_id(),
                &section1.id.as_id(),
                MoveSectionDirection::Up,
            )
            .await
            .is_err());

        // move the last section down, should return error
        assert!(service
            .move_section(
                &page_id.as_id(),
                &section0.id.as_id(),
                MoveSectionDirection::Down,
            )
            .await
            .is_err());
    }
}
