mod types;
use async_trait::async_trait;
use juniper::ID;
pub use types::*;

use crate::schema::Result;

#[async_trait]
pub trait PageService: Send + Sync {
    async fn convert_thread_to_page(&self, author_id: &ID, thread_id: &ID) -> Result<ID>;

    async fn generate_page_title(&self, id: &ID) -> Result<String>;
    async fn generate_page_summary(&self, id: &ID) -> Result<String>;

    async fn get(&self, id: &ID) -> Result<Page>;
    async fn list(
        &self,
        ids: Option<&[ID]>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<Page>>;
    async fn delete(&self, id: &ID) -> Result<()>;

    async fn update_section(&self, input: &UpdateSectionInput) -> Result<Section>;
    async fn reorder_section(&self, input: &ReorderSectionInput) -> Result<()>;

    async fn get_section(&self, page_id: &ID, id: &ID) -> Result<Section>;
    async fn delete_section(&self, page_id: &ID, id: &ID) -> Result<()>;
}
