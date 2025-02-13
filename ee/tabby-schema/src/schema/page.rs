mod types;
use async_trait::async_trait;
use futures::stream::BoxStream;
use juniper::ID;
pub use types::*;

use crate::{policy::AccessPolicy, schema::Result, thread::Message};

pub type ThreadToPageRunStream = BoxStream<'static, Result<PageRunItem>>;

#[async_trait]
pub trait PageService: Send + Sync {
    async fn convert_thread_to_page(
        &self,
        policy: &AccessPolicy,
        author_id: &ID,
        thread_id: &ID,
    ) -> Result<ThreadToPageRunStream>;

    async fn generate_page_title(
        &self,
        policy: &AccessPolicy,
        page_id: ID,
        messages: &Vec<Message>,
    ) -> Result<String>;

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
    async fn delete_section(&self, id: &ID) -> Result<()>;
}
