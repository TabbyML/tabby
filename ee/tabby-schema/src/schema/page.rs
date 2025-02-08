mod types;
use async_trait::async_trait;
use futures::stream::BoxStream;
use juniper::ID;
pub use types::*;

use crate::schema::Result;

pub type ThreadToPageRunStream = BoxStream<'static, Result<PageRunItem>>;

#[async_trait]
pub trait PageService: Send + Sync {
    async fn convert_thread_to_page(
        &self,
        author_id: &ID,
        thread_id: &ID,
    ) -> Result<ThreadToPageRunStream>;

    async fn generate_page_title(&self, id: &ID) -> Result<String>;
    async fn generate_page_content(&self, id: &ID) -> Result<String>;

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
    async fn add_section(&self, input: &AddPageSectionInput) -> Result<ID>;
    async fn get_section(&self, id: &ID) -> Result<Section>;
    async fn delete_section(&self, id: &ID) -> Result<()>;
}
