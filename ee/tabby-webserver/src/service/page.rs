use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use juniper::ID;
use tabby_db::DbConn;
use tabby_schema::{
    auth::AuthenticationService,
    page::{
        AddPageSectionInput, Page, PageCompleted, PageContentCompleted, PageCreated, PageRunItem,
        PageSection, PageSectionContentCompleted, PageSectionContentDelta, PageSectionsCreated,
        PageService, Section, ThreadToPageRunStream,
    },
    thread::{Message, ThreadService},
    AsID, AsRowid, CoreError, Result,
};

use super::{answer::AnswerService, graphql_pagination_to_filter};

struct PageServiceImpl {
    db: DbConn,
    _auth: Arc<dyn AuthenticationService>,
    thread: Arc<dyn ThreadService>,
    answer: Arc<AnswerService>,
}

pub fn create(
    db: DbConn,
    auth: Arc<dyn AuthenticationService>,
    thread: Arc<dyn ThreadService>,
    answer: Arc<AnswerService>,
) -> impl PageService {
    PageServiceImpl {
        db,
        thread,
        _auth: auth,
        answer,
    }
}

#[async_trait]
impl PageService for PageServiceImpl {
    async fn convert_thread_to_page(
        &self,
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

        let title = self.generate_page_title(page_id.clone(), &messages).await?;
        let answer = self.answer.clone();
        let db = self.db.clone();

        let author_id = author_id.clone();
        let s = async_stream::stream! {
            yield Ok(PageRunItem::PageCreated(PageCreated {
                id: page_id.clone(),
                author_id: author_id.clone(),
                title,
            }));

            let content_stream = answer.generate_page_content(&messages).await?;
            for await delta in content_stream {
                let delta = delta?;
                db.append_page_content(page_id.as_rowid()?, &delta.delta).await?;
                yield Ok(PageRunItem::PageContentDelta(delta));
            }

            yield Ok(PageRunItem::PageContentCompleted(PageContentCompleted {
                id: page_id.clone(),
            }));

            let sections = answer.generate_page_sections(&messages).await?;
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
                let content_stream = answer.generate_page_section_content(&messages, &section_titles, &section.title).await?;
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

    async fn generate_page_title(&self, page_id: ID, messages: &Vec<Message>) -> Result<String> {
        let title = self.answer.generate_page_title(messages).await?;

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
        //TODO: generate section content
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
