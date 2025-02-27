mod prompt_tools;

use std::sync::Arc;

use anyhow::anyhow;
use async_stream::stream;
use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};
use juniper::ID;
use prompt_tools::{
    prompt_page_content, prompt_page_section_content, prompt_page_section_titles, prompt_page_title,
};
use tabby_db::DbConn;
use tabby_inference::ChatCompletionStream;
use tabby_schema::{
    context::ContextService,
    page::{
        CreatePageSectionRunInput, MoveSectionDirection, Page, PageCompleted, PageContentCompleted,
        PageContentDelta, PageCreated, PageRunItem, PageSection, PageSectionContentCompleted,
        PageSectionContentDelta, PageSectionsCreated, PageService, Section, SectionRunItem,
        SectionRunStream, ThreadToPageRunStream,
    },
    policy::AccessPolicy,
    thread::{Message, MessageAttachment, ThreadService},
    AsID, AsRowid, CoreError, Result,
};

use super::graphql_pagination_to_filter;
use crate::service::utils::{
    convert_messages_to_chat_completion_request, convert_user_message_to_chat_completion_request,
    prompt::{request_llm_stream, request_llm_with_message, transform_line_items},
};

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

        let s = stream! {
            yield Ok(PageRunItem::PageCreated(PageCreated {
                id: page_id.clone(),
                author_id: author_id.clone(),
                title: title.clone(),
            }));

            let sections = generate_page_sections(
                3,
                chat.clone(),
                context.clone(),
                &policy,
                "",
                &title,
                &vec![],
                &messages,
            ).await?;
            let mut page_sections = Vec::new();
            for section_title in sections {
                let section = db.create_page_section(page_id.as_rowid()?, &section_title).await?;
                page_sections.push(PageSection {
                    id: section.0.as_id(),
                    title: section_title.to_owned(),
                    position: section.1 as i32,
                });
            }

            yield Ok(PageRunItem::PageSectionsCreated(PageSectionsCreated {
                sections: page_sections.clone(),
            }));

            let content_stream = generate_page_content(chat.clone(), context.clone(), &policy, &title, &messages).await?;
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

            for section in page_sections {
                let section_id = section.id.clone();

                let existed_sections = db
                    .list_page_sections(page_id.as_rowid()?, None, None, false)
                    .await?
                    .into_iter()
                    .map(Into::into)
                    .collect();
                let content_stream = generate_page_section_content(
                    chat.clone(),
                    context.clone(),
                    &policy,
                    &messages,
                    &title,
                    &existed_sections,
                    &section.title,
                ).await?;
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

    async fn append_section(
        &self,
        policy: &AccessPolicy,
        input: &CreatePageSectionRunInput,
    ) -> Result<SectionRunStream> {
        let current_sections = self
            .list_sections(&input.page_id, None, None, None, None)
            .await?;

        let chat = self.chat.clone();
        let context = self.context.clone();
        let db = self.db.clone();
        let policy = policy.clone();
        let page_id = input.page_id.as_rowid()?;
        let current_title = input.title.clone();

        let s = stream! {
            let page_title = db
                .get_page_title(page_id)
                .await?;
            let title = generate_page_sections(
                1,
                chat.clone(),
                context.clone(),
                &policy,
                &current_title,
                &page_title,
                &current_sections,
                &vec![],
            )
            .await?
            .first()
            .ok_or_else(|| CoreError::Other(anyhow!("failed to generate section title")))?
            .to_owned();

            let (section_id, position) = db
                .create_page_section(page_id, &title)
                .await?;
            yield Ok(SectionRunItem::PageSectionCreated (PageSection {
                id: section_id.as_id(),
                title: title.clone(),
                position: position as i32,
            }));

            let content_stream = generate_page_section_content(
                chat.clone(),
                context.clone(),
                &policy,
                &vec![],
                &page_title,
                &current_sections,
                &title,
            )
            .await?;
            for await delta in content_stream {
                let delta = delta?;
                db
                    .append_page_section_content(section_id, &delta)
                    .await
                    .unwrap();
                yield Ok(SectionRunItem::PageSectionContentDelta(PageSectionContentDelta{
                    id: section_id.as_id(),
                    delta
                }));
            }

            yield Ok(SectionRunItem::PageSectionContentCompleted(PageSectionContentCompleted {
                id: section_id.as_id(),
            }));
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
}

impl PageServiceImpl {
    async fn generate_page_title(
        &self,
        policy: &AccessPolicy,
        page_id: ID,
        messages: &Vec<Message>,
    ) -> Result<String> {
        let helper = self.context.read(Some(policy)).await?.helper();
        let mut messages = convert_messages_to_chat_completion_request(None, &helper, messages)?;

        let user_message = convert_user_message_to_chat_completion_request(
            &helper,
            prompt_page_title(),
            &MessageAttachment::default(),
            None,
        );
        messages.push(user_message);

        let title = request_llm_with_message(self.chat.clone(), messages).await?;
        let title = trim_title(title.as_ref());

        self.db
            .update_page_title(page_id.as_rowid()?, title)
            .await?;
        Ok(title.to_owned())
    }
}

fn trim_title(title: &str) -> &str {
    title.trim_matches(&['"', '#', ' ', '-'][..])
}

async fn generate_page_content(
    chat: Arc<dyn ChatCompletionStream>,
    context: Arc<dyn ContextService>,
    policy: &AccessPolicy,
    title: &str,
    messages: &Vec<Message>,
) -> tabby_schema::Result<BoxStream<'static, tabby_schema::Result<String>>> {
    let helper = context.read(Some(policy)).await?.helper();
    let mut messages = convert_messages_to_chat_completion_request(None, &helper, messages)?;

    let user_message = convert_user_message_to_chat_completion_request(
        &helper,
        prompt_page_content(title).as_str(),
        &MessageAttachment::default(),
        None,
    );
    messages.push(user_message);

    Ok(request_llm_stream(chat.clone(), messages).await)
}

pub async fn generate_page_sections(
    count: usize,
    chat: Arc<dyn ChatCompletionStream>,
    context: Arc<dyn ContextService>,
    policy: &AccessPolicy,
    new_section: &str,
    title: &str,
    sections: &Vec<Section>,
    messages: &Vec<Message>,
) -> anyhow::Result<Vec<String>> {
    let helper = context.read(Some(policy)).await?.helper();
    let mut messages = convert_messages_to_chat_completion_request(None, &helper, messages)?;

    let sections = sections
        .iter()
        .map(|x| format!("## {}\n\n{}", x.title, x.content))
        .collect::<Vec<_>>()
        .join("\n");
    let page = format!(
        r#"# {title}

{sections}
        "#,
    );

    let user_message = convert_user_message_to_chat_completion_request(
        &helper,
        prompt_page_section_titles(count, &page, new_section).as_str(),
        &MessageAttachment::default(),
        None,
    );
    messages.push(user_message);

    let titles = request_llm_with_message(chat.clone(), messages).await?;
    Ok(transform_line_items(&titles)
        .into_iter()
        .map(|x| trim_title(&x).to_owned())
        .collect())
}

pub async fn generate_page_section_content(
    chat: Arc<dyn ChatCompletionStream>,
    context: Arc<dyn ContextService>,
    policy: &AccessPolicy,
    messages: &Vec<Message>,
    title: &str,
    sections: &Vec<Section>,
    current_section: &str,
) -> tabby_schema::Result<BoxStream<'static, tabby_schema::Result<String>>> {
    let helper = context.read(Some(policy)).await?.helper();
    let mut messages = convert_messages_to_chat_completion_request(None, &helper, messages)?;

    let page = format!(
        r#"
## {}

{}"#,
        title,
        sections
            .iter()
            .map(|x| format!("## {}\n\n{}", x.title, x.content))
            .collect::<Vec<_>>()
            .join("\n")
    );

    let user_message = convert_user_message_to_chat_completion_request(
        &helper,
        prompt_page_section_content(&page, current_section).as_str(),
        &MessageAttachment::default(),
        None,
    );
    messages.push(user_message);

    Ok(request_llm_stream(chat.clone(), messages).await)
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_move_section() {
        use tabby_db::DbConn;
        use tabby_schema::page::MoveSectionDirection;

        use super::*;
        use crate::{
            answer::testutils::{FakeChatCompletionStream, FakeContextService},
            service::thread,
        };

        let db = DbConn::new_in_memory().await.unwrap();
        let user_id = db
            .create_user("test@example.com".into(), None, true, None)
            .await
            .unwrap();
        let page_id = db.create_page(user_id).await.unwrap();
        let section0 = db.create_page_section(page_id, "Section 0").await.unwrap();
        assert_eq!(section0.1, 0);
        let section1 = db.create_page_section(page_id, "Section 1").await.unwrap();
        assert_eq!(section1.1, 1);
        let section2 = db.create_page_section(page_id, "Section 2").await.unwrap();
        assert_eq!(section2.1, 2);

        let chat: Arc<dyn ChatCompletionStream> = Arc::new(FakeChatCompletionStream {
            return_error: false,
        });
        let context: Arc<dyn ContextService> = Arc::new(FakeContextService);
        let thread = Arc::new(thread::create(db.clone(), None, None, context.clone()));
        let context: Arc<dyn ContextService> = Arc::new(FakeContextService);
        let service = create(db, chat, thread, context);

        // move down
        service
            .move_section(
                &page_id.as_id(),
                &section0.0.as_id(),
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
                &section2.0.as_id(),
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
                &section1.0.as_id(),
                MoveSectionDirection::Up,
            )
            .await
            .is_err());

        // move the last section down, should return error
        assert!(service
            .move_section(
                &page_id.as_id(),
                &section0.0.as_id(),
                MoveSectionDirection::Down,
            )
            .await
            .is_err());
    }
}
