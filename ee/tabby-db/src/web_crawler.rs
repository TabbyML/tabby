use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::{prelude::FromRow, query};
use tabby_db_macros::query_paged_as;

use crate::DbConn;

#[derive(FromRow)]
pub struct WebCrawlerUrlDAO {
    pub id: i64,
    pub url: String,
    pub created_at: DateTime<Utc>,
}

#[allow(unused)]
#[derive(FromRow)]
pub struct WebDocumentDAO {
    pub id: i64,
    pub name: String,
    pub url: String,
    pub is_preset: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
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
            ["id", "url", "created_at" as "created_at!: DateTime<Utc>"],
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

    pub async fn list_web_document(
        &self,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
        is_preset: Option<bool>,
    ) -> Result<Vec<WebCrawlerUrlDAO>> {
        let condition = is_preset.map(|is_preset| format!("is_preset={}", is_preset));

        let urls = query_paged_as!(
            WebDocumentDAO,
            "web_documents",
            ["id", "name", "url", "is_preset", "created_at" as "created_at!: DateTime<Utc>", "updated_at" as "updated_at!: DateTime<Utc>"],
            limit,
            skip_id,
            backwards,
            condition
        ).fetch_all(&self.pool)
            .await?;

        Ok(urls)
    }

    pub async fn create_web_document(
        &self,
        name: String,
        url: String,
        is_preset: bool,
    ) -> Result<i64> {
        let res = query!(
            "INSERT INTO web_documents(name, url, is_preset) VALUES (?,?,?);",
            name,
            url,
            is_preset
        )
        .execute(&self.pool)
        .await?;

        Ok(res.last_insert_rowid())
    }

    pub async fn delete_web_document(&self, id: i64) -> Result<()> {
        query!("DELETE FROM web_documents WHERE id = ?;", id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}
