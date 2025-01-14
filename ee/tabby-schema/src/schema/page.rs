mod types;
pub use types::*;

use async_trait::async_trait;
use juniper::ID;

use crate::schema::Result;

#[async_trait]
pub trait PageService: Send + Sync {
    async fn convert_thread_to_page(&self, author_id: &ID, thread_id: &ID) -> Result<ID>;

    async fn delete(&self, id: &ID) -> Result<()>;
    async fn generate_page_title(&self, id: &ID) -> Result<String>;
    async fn generate_page_summary(&self, id: &ID) -> Result<String>;

    async fn list(
        &self,
        ids: Option<&[ID]>,
        after: Option<String>,
        before: Option<String>,
        first: Option<usize>,
        last: Option<usize>,
    ) -> Result<Vec<Page>>;
}
