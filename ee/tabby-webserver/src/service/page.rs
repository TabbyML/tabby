mod prompt_tools;

use std::sync::Arc;

use anyhow::anyhow;
use async_openai_alt::types::ChatCompletionRequestMessage;
use async_stream::stream;
use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};
use juniper::ID;
use prompt_tools::{
    prompt_page_content, prompt_page_section_content, prompt_page_section_titles, prompt_page_title,
};
use tabby_common::config::PageConfig;
use tabby_db::DbConn;
use tabby_inference::ChatCompletionStream;
use tabby_schema::{
    context::ContextService,
    page::{
        CreatePageRunInput, CreatePageSectionRunInput, MoveSectionDirection, Page, PageCompleted,
        PageContentCompleted, PageContentDelta, PageCreated, PageRunItem, PageRunStream,
        PageSection, PageSectionAttachmentCode, PageSectionAttachmentCodeFileList,
        PageSectionContentCompleted, PageSectionContentDelta, PageSectionsCreated, PageService,
        SectionAttachment, SectionRunItem, SectionRunStream, ThreadToPageRunStream,
    },
    policy::AccessPolicy,
    retrieval::AttachmentCodeFileList,
    thread::{CodeQueryInput, Message, ThreadService},
    AsID, AsRowid, CoreError, Result,
};
use tracing::error;

use super::{graphql_pagination_to_filter, retrieval::RetrievalService, utils::get_source_id};
use crate::service::utils::{
    convert_messages_to_chat_completion_request, convert_user_message_to_chat_completion_request,
    prompt::{request_llm_stream, request_llm_with_message, transform_line_items, trim_title},
};

struct PageServiceImpl {
    config: PageConfig,
    db: DbConn,
    chat: Arc<dyn ChatCompletionStream>,
    thread: Arc<dyn ThreadService>,
    context: Arc<dyn ContextService>,
    retrieval: Arc<RetrievalService>,
}

pub fn create(
    config: PageConfig,
    db: DbConn,
    chat: Arc<dyn ChatCompletionStream>,
    thread: Arc<dyn ThreadService>,
    context: Arc<dyn ContextService>,
    retrieval: Arc<RetrievalService>,
) -> impl PageService {
    PageServiceImpl {
        config,
        db,
        chat,
        thread,
        context,
        retrieval,
    }
}

#[async_trait]
impl PageService for PageServiceImpl {
    async fn convert_thread_to_page(
        &self,
        policy: &AccessPolicy,
        author_id: &ID,
        thread_id: &ID,
    ) -> Result<ThreadToPageRunStream> {
        let _thread = self
            .thread
            .get(thread_id)
            .await?
            .ok_or_else(|| CoreError::NotFound("Thread not found"))?;

        let thread_messages = self
            .thread
            .list_thread_messages(thread_id, None, None, None, None)
            .await?;

        self.page_run(policy, author_id, None, None, Some(&thread_messages))
            .await
    }

    async fn create_run(
        &self,
        policy: &AccessPolicy,
        author_id: &ID,
        input: &CreatePageRunInput,
    ) -> Result<PageRunStream> {
        self.page_run(
            policy,
            author_id,
            Some(&input.title_prompt),
            input.code_query.as_ref(),
            None,
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

        let new_section_title = generate_page_sections(
            1,
            self.chat.clone(),
            self.context.clone(),
            policy,
            Some(&new_section_prompt),
            &page_title,
            &existing_page_sections,
            None,
        )
        .await?
        .first()
        .ok_or_else(|| CoreError::Other(anyhow!("failed to generate section title")))?
        .to_owned();

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

        let s = stream! {
            yield Ok(SectionRunItem::PageSectionCreated(section.into()));

            let mut attachments_stream = generate_section_with_attachments(
                policy,
                section_id,
                new_section_title,
                Some(new_section_prompt),
                &existing_page_sections,
                &page_title,
                code_source_id,
                None,
                db,
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

        Ok(sections.into_iter().map(Into::into).collect())
    }

    async fn get_section(&self, id: &ID) -> Result<PageSection> {
        let section = self
            .db
            .get_page_section(id.as_rowid()?)
            .await?
            .ok_or_else(|| CoreError::NotFound("Section not found"))?;
        Ok(section.into())
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
        code_query: Option<&CodeQueryInput>,
        thread_messages: Option<&[Message]>,
    ) -> Result<PageRunStream> {
        let code_source_id = if let Some(code_query) = code_query {
            get_source_id(self.context.clone(), policy, code_query).await
        } else {
            // use the first message source id for page
            thread_messages.and_then(|messages| {
                messages
                    .iter()
                    .find_map(|m| m.code_source_id.as_deref())
                    .map(ToOwned::to_owned)
            })
        };
        let thread_messages = thread_messages.map(ToOwned::to_owned);

        let page_id = self
            .db
            .create_page(author_id.as_rowid()?, code_source_id.clone())
            .await?
            .as_id();

        let page_title = self
            .generate_page_title(policy, &page_id, title_prompt, thread_messages.as_deref())
            .await?;

        let db = self.db.clone();
        let policy = policy.clone();
        let chat = self.chat.clone();
        let context = self.context.clone();
        let author_id = author_id.clone();
        let retrieval = self.retrieval.clone();
        let config = self.config.clone();

        let s = stream! {
            yield Ok(PageRunItem::PageCreated(PageCreated {
                id: page_id.clone(),
                author_id: author_id.clone(),
                title: page_title.clone(),
            }));

            let page_section_titles = generate_page_sections(
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
                page_sections.push(section.into());
            }

            yield Ok(PageRunItem::PageSectionsCreated(PageSectionsCreated {
                sections: page_sections.clone(),
            }));

            let content_stream = generate_page_content(
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
            }));

            for (i, section) in page_sections.iter().enumerate() {
                let existed_sections: Vec<_> = db
                    .list_page_sections(page_id.as_rowid()?, None, None, false)
                    .await?
                    .into_iter()
                    .map(Into::into)
                    // Only take the sections before the current section
                    .take(i)
                    .collect();

                let attachments_stream = generate_section_with_attachments(
                    policy.clone(),
                    section.id.as_rowid()?,
                    section.title.clone(),
                    None,
                    &existed_sections,
                    &page_title,
                    code_source_id.clone(),
                    thread_messages.as_deref(),
                    db.clone(),
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
        policy: &AccessPolicy,
        page_id: &ID,
        title_prompt: Option<&str>,
        thread_messages: Option<&[Message]>,
    ) -> Result<String> {
        let messages = build_chat_messages(
            self.context.clone(),
            policy,
            thread_messages,
            prompt_page_title(title_prompt).as_str(),
            &SectionAttachment::default(),
        )
        .await?;

        let title = request_llm_with_message(self.chat.clone(), messages).await?;
        let title = trim_title(title.as_str());

        self.db
            .update_page_title(page_id.as_rowid()?, title)
            .await?;
        Ok(title.to_owned())
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

async fn generate_page_content(
    chat: Arc<dyn ChatCompletionStream>,
    context: Arc<dyn ContextService>,
    policy: &AccessPolicy,
    page_title: &str,
    page_section_titles: &[String],
    thread_messages: Option<&[Message]>,
) -> tabby_schema::Result<BoxStream<'static, tabby_schema::Result<String>>> {
    let messages = build_chat_messages(
        context,
        policy,
        thread_messages,
        prompt_page_content(page_title, page_section_titles).as_str(),
        &SectionAttachment::default(),
    )
    .await?;

    Ok(request_llm_stream(chat.clone(), messages).await)
}

async fn generate_section_with_attachments(
    policy: AccessPolicy,
    section_id: i64,
    section_title: String,
    section_prompt: Option<String>,
    existing_sections: &[PageSection],
    page_title: &str,
    code_source_id: Option<String>,
    thread_messages: Option<&[Message]>,
    db: DbConn,
    chat: Arc<dyn ChatCompletionStream>,
    context: Arc<dyn ContextService>,
    retrieval: Arc<RetrievalService>,
    config: PageConfig,
) -> Result<BoxStream<'static, Result<SectionRunItem>>> {
    let policy = policy.clone();
    let existing_sections = existing_sections.to_vec();
    let page_title = page_title.to_string();
    let thread_messages = thread_messages.map(ToOwned::to_owned);
    let context_info_helper = context.read(Some(&policy)).await?.helper();

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
                    content: query_content,
                    ..Default::default()
                },
                &config.code_search_params,
                None,
            ).await?;

            attachment.code = hits.iter().map(|x| x.doc.clone().into()).collect::<Vec<_>>();

            if !hits.is_empty() {
                let hits = hits.into_iter().map(|x| x.into()).collect::<Vec<_>>();
                db.update_page_section_code_attachments(section_id, &attachment.code.iter().map(|c| c.into()).collect::<Vec<_>>()).await?;
                yield Ok(SectionRunItem::PageSectionAttachmentCode(
                    PageSectionAttachmentCode {
                        id: section_id.as_id(),
                        codes: hits,
                    }
                ));
            }
        }

        // Generate section content
        let content_stream = generate_page_section_content(
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
        }));
    };

    Ok(stream.boxed())
}

pub async fn generate_page_sections(
    count: usize,
    chat: Arc<dyn ChatCompletionStream>,
    context: Arc<dyn ContextService>,
    policy: &AccessPolicy,
    new_section_prompt: Option<&str>,
    page_title: &str,
    page_sections: &[PageSection],
    thread_messages: Option<&[Message]>,
) -> anyhow::Result<Vec<String>> {
    let messages = build_chat_messages(
        context,
        policy,
        thread_messages,
        prompt_page_section_titles(count, page_title, page_sections, new_section_prompt).as_str(),
        &SectionAttachment::default(),
    )
    .await?;

    let titles = request_llm_with_message(chat.clone(), messages).await?;
    Ok(transform_line_items(&titles)
        .into_iter()
        .map(|x| trim_title(x.as_str()).to_owned())
        .collect())
}

pub async fn generate_page_section_content(
    chat: Arc<dyn ChatCompletionStream>,
    context: Arc<dyn ContextService>,
    policy: &AccessPolicy,
    thread_messages: Option<&[Message]>,
    page_title: &str,
    page_sections: &[PageSection],
    new_section_title: &str,
    new_section_attachment: &SectionAttachment,
) -> tabby_schema::Result<BoxStream<'static, tabby_schema::Result<String>>> {
    let messages = build_chat_messages(
        context,
        policy,
        thread_messages,
        prompt_page_section_content(page_title, page_sections, new_section_title).as_str(),
        new_section_attachment,
    )
    .await?;

    Ok(request_llm_stream(chat.clone(), messages).await)
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_move_section() {
        use tabby_common::api::structured_doc::DocSearch;
        use tabby_db::DbConn;
        use tabby_schema::page::MoveSectionDirection;

        use super::*;
        use crate::{
            answer::testutils::{
                make_repository_service, FakeChatCompletionStream, FakeCodeSearch,
                FakeContextService, FakeDocSearch,
            },
            retrieval,
            service::thread,
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

        let retrieval = Arc::new(retrieval::create(
            code.clone(),
            doc.clone(),
            serper,
            repo_service.clone(),
        ));
        let service = create(PageConfig::default(), db, chat, thread, context, retrieval);

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
