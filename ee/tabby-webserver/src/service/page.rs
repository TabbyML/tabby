mod prompt_tools;

use std::sync::Arc;

use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};
use juniper::ID;
use prompt_tools::{
    pipeline_page_sections, pipeline_page_title, prompt_page_content, prompt_page_section_content,
};
use tabby_db::DbConn;
use tabby_inference::ChatCompletionStream;
use tabby_schema::{
    context::ContextService,
    page::{
        AddPageSectionInput, Page, PageCompleted, PageContentCompleted, PageContentDelta,
        PageCreated, PageRunItem, PageSection, PageSectionContentCompleted,
        PageSectionContentDelta, PageSectionsCreated, PageService, Section, ThreadToPageRunStream,
    },
    policy::AccessPolicy,
    thread::{Message, ThreadService},
    AsID, AsRowid, CoreError, Result,
};

use super::graphql_pagination_to_filter;
use crate::service::utils::prompt::request_llm_stream;

struct PageServiceImpl {
    db: DbConn,
    chat: Arc<dyn ChatCompletionStream>,
    thread: Arc<dyn ThreadService>,
    context: Arc<dyn ContextService>,
}

pub fn create(
    db: DbConn,
    chat: Arc<dyn ChatCompletionStream>,
    thread: Arc<dyn ThreadService>,
    context: Arc<dyn ContextService>,
) -> impl PageService {
    PageServiceImpl {
        db,
        chat,
        thread,
        context,
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
        let page_id = self.db.create_page(author_id.as_rowid()?).await?.as_id();

        let messages = self
            .thread
            .list_thread_messages(thread_id, None, None, None, None)
            .await?;

        let title = self
            .generate_page_title(policy, page_id.clone(), &messages)
            .await?;
        let db = self.db.clone();
        let policy = policy.clone();
        let chat = self.chat.clone();
        let context = self.context.clone();

        let author_id = author_id.clone();
        let s = async_stream::stream! {
            yield Ok(PageRunItem::PageCreated(PageCreated {
                id: page_id.clone(),
                author_id: author_id.clone(),
                title,
            }));

            let content_stream = generate_page_content(chat.clone(), context.clone(), &policy, &messages).await?;
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

            let sections = generate_page_sections(chat.clone(), context.clone(), &policy, &messages).await?;
            let mut page_sections = Vec::new();
            for section_title in sections {
                let section = db.create_page_section(page_id.as_rowid()?, &section_title).await?;
                page_sections.push(PageSection {
                    id: section.as_id(),
                    title: section_title,
                });
            }

            yield Ok(PageRunItem::PageSectionsCreated(PageSectionsCreated {
                sections: page_sections.clone(),
            }));

            let section_titles = page_sections.iter().map(|x| x.title.clone()).collect();
            for section in page_sections {
                let section_id = section.id.clone();
                let content_stream = generate_page_section_content(chat.clone(), context.clone(), &policy, &messages, &section_titles, &section.title).await?;
                for await delta in content_stream {
                    let delta = delta?;
                    db.append_page_section_content(section_id.clone().as_rowid()?, &delta).await?;
                    yield Ok(PageRunItem::PageSectionContentDelta(PageSectionContentDelta {
                        id: section_id.clone(),
                        delta
                    }));
                }

                yield Ok(PageRunItem::PageSectionContentCompleted(PageSectionContentCompleted {
                    id: section_id,
                }));
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
        page_id: ID,
        messages: &Vec<Message>,
    ) -> Result<String> {
        let context_info = self.context.read(Some(policy)).await?;
        let context_info_helper = context_info.helper();
        let content = messages
            .iter()
            .map(|x| x.content.clone())
            .collect::<Vec<_>>()
            .join("\n");
        let context = context_info_helper.rewrite_tag(&content);
        let title = pipeline_page_title(self.chat.clone(), &context, &content).await?;

        self.db
            .update_page_title(page_id.as_rowid()?, &title)
            .await?;
        Ok(title)
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
    ) -> Result<Vec<Section>> {
        let (limit, skip_id, backwards) = graphql_pagination_to_filter(after, before, first, last)?;

        let sections = self
            .db
            .list_page_sections(page_id.as_rowid()?, limit, skip_id, backwards)
            .await?;

        Ok(sections.into_iter().map(Into::into).collect())
    }

    async fn get_section(&self, id: &ID) -> Result<Section> {
        let section = self
            .db
            .get_page_section(id.as_rowid()?)
            .await?
            .ok_or_else(|| CoreError::NotFound("Section not found"))?;
        Ok(section.into())
    }

    async fn add_section(&self, input: &AddPageSectionInput) -> Result<ID> {
        //TODO(kweizh): generate section content
        let section = self
            .db
            .create_page_section(input.page_id.as_rowid()?, &input.title)
            .await?;
        Ok(section.as_id())
    }

    async fn delete_section(&self, id: &ID) -> Result<()> {
        self.db.delete_page_section(id.as_rowid()?).await?;
        Ok(())
    }
}

async fn generate_page_content(
    chat: Arc<dyn ChatCompletionStream>,
    context: Arc<dyn ContextService>,
    policy: &AccessPolicy,
    messages: &Vec<Message>,
) -> tabby_schema::Result<BoxStream<'static, tabby_schema::Result<String>>> {
    let context_info = context.read(Some(policy)).await?;
    let context_info_helper = context_info.helper();

    let content = messages
        .iter()
        .map(|x| x.content.clone())
        .collect::<Vec<_>>()
        .join("\n");

    let context = context_info_helper.rewrite_tag(&content);

    let prompt = prompt_page_content(&context, &content);
    Ok(request_llm_stream(chat, prompt).await)
}

pub async fn generate_page_sections(
    chat: Arc<dyn ChatCompletionStream>,
    context: Arc<dyn ContextService>,
    policy: &AccessPolicy,
    messages: &Vec<Message>,
) -> anyhow::Result<Vec<String>> {
    let context_info = context.read(Some(policy)).await?;
    let context_info_helper = context_info.helper();
    let content = messages
        .iter()
        .map(|x| x.content.clone())
        .collect::<Vec<_>>()
        .join("\n");
    let context = context_info_helper.rewrite_tag(&content);
    pipeline_page_sections(chat.clone(), &context, &content).await
}

pub async fn generate_page_section_content(
    chat: Arc<dyn ChatCompletionStream>,
    context: Arc<dyn ContextService>,
    policy: &AccessPolicy,
    messages: &Vec<Message>,
    sections: &Vec<String>,
    current_section: &str,
) -> tabby_schema::Result<BoxStream<'static, tabby_schema::Result<String>>> {
    let context_info = context.read(Some(policy)).await?;
    let context_info_helper = context_info.helper();

    let content = messages
        .iter()
        .map(|x| x.content.clone())
        .collect::<Vec<_>>()
        .join("\n");

    let context = context_info_helper.rewrite_tag(&content);

    let sections = sections
        .iter()
        .map(|s| format!("- {}", s))
        .collect::<Vec<_>>()
        .join("\n");

    let current = current_section.to_owned();

    let prompt = prompt_page_section_content(&context, &content, &sections, &current);
    Ok(request_llm_stream(chat.clone(), prompt).await)
}
