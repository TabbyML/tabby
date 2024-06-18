use anyhow::Result;
use sqlx::{prelude::FromRow, query};
use tabby_db_macros::query_paged_as;

use crate::{DateTimeUtc, DbConn};

#[derive(FromRow)]
pub struct WebCrawlerUrlDAO {
    pub id: i64,
    pub url: String,
    pub created_at: DateTimeUtc,
}

impl DbConn {
    pub async fn list_web_crawler_urls(
        &self,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<WebCrawlerUrlDAO>> {
        let urls = query_paged_as!(
            WebCrawlerUrlDAO,
            "web_crawler_urls",
            ["id", "url", "created_at"],
            limit,
            skip_id,
            backwards
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(urls)
    }

    pub async fn create_web_crawler_url(&self, url: String) -> Result<i64> {
        let res = query!("INSERT INTO web_crawler_urls(url) VALUES (?);", url)
            .execute(&self.pool)
            .await?;

        Ok(res.last_insert_rowid())
    }

    pub async fn delete_web_crawler_url(&self, id: i64) -> Result<()> {
        query!("DELETE FROM web_crawler_urls WHERE id = ?;", id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}
