mod types;

use async_trait::async_trait;
use futures::stream::BoxStream;
use juniper::ID;
pub use types::*;

use crate::{policy::AccessPolicy, schema::Result};

pub type ThreadToPageRunStream = BoxStream<'static, Result<PageRunItem>>;
pub type PageRunStream = BoxStream<'static, Result<PageRunItem>>;
pub type SectionRunStream = BoxStream<'static, Result<SectionRunItem>>;

#[async_trait]
pub trait PageService: Send + Sync {
    fn source_id(&self) -> String {
        "page".into()
    }

    async fn convert_thread_to_page(
        &self,
        policy: &AccessPolicy,
        author_id: &ID,
        input: &CreateThreadToPageRunInput,
    ) -> Result<ThreadToPageRunStream>;

    async fn create_run(
        &self,
        policy: &AccessPolicy,
        author_id: &ID,
        input: &CreatePageRunInput,
    ) -> Result<PageRunStream>;

    async fn list(
        &self,
        ids: Option<&[ID]>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<Page>>;
    async fn get(&self, id: &ID) -> Result<Page>;
    async fn update_title(&self, id: &ID, title: &str) -> Result<()>;
    async fn update_content(&self, id: &ID, content: &str) -> Result<()>;
    async fn delete(&self, id: &ID) -> Result<()>;

    async fn list_sections(
        &self,
        page_id: &ID,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<PageSection>>;
    async fn get_section(&self, id: &ID) -> Result<PageSection>;
    async fn move_section(
        &self,
        page_id: &ID,
        id: &ID,
        direction: MoveSectionDirection,
    ) -> Result<()>;
    async fn append_section(
        &self,
        policy: &AccessPolicy,
        input: &CreatePageSectionRunInput,
    ) -> Result<SectionRunStream>;
    async fn update_section_title(&self, id: &ID, title: &str) -> Result<()>;
    async fn update_section_content(&self, id: &ID, content: &str) -> Result<()>;

    async fn delete_section(&self, id: &ID) -> Result<()>;
}
