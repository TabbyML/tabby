mod types;
use async_trait::async_trait;
use futures::stream::BoxStream;
use juniper::ID;
pub use types::*;

use crate::{policy::AccessPolicy, schema::Result};

pub type ThreadToPageRunStream = BoxStream<'static, Result<PageRunItem>>;
pub type SectionRunStream = BoxStream<'static, Result<SectionRunItem>>;

#[async_trait]
pub trait PageService: Send + Sync {
    async fn convert_thread_to_page(
        &self,
        policy: &AccessPolicy,
        author_id: &ID,
        thread_id: &ID,
    ) -> Result<ThreadToPageRunStream>;

    async fn list(
        &self,
        ids: Option<&[ID]>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<Page>>;
    async fn get(&self, id: &ID) -> Result<Page>;
    async fn delete(&self, id: &ID) -> Result<()>;

    async fn list_sections(
        &self,
        page_id: &ID,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<Section>>;
    async fn get_section(&self, id: &ID) -> Result<Section>;
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
    async fn delete_section(&self, id: &ID) -> Result<()>;
}
