use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use sqlx::{prelude::FromRow, query, query_as, query_scalar};
use tabby_db_macros::query_paged_as;

use crate::DbConn;

#[derive(FromRow)]
pub struct WebCrawlerUrlDAO {
    pub id: i64,
    pub url: String,
    pub web_name: String,
    pub active: bool,
    pub is_preset: bool,
    pub created_at: DateTime<Utc>,
}

impl DbConn {
    pub async fn list_web_crawler_urls(
        &self,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
        active: Option<bool>,
        is_preset: Option<bool>,
    ) -> Result<Vec<WebCrawlerUrlDAO>> {
        let mut conditions = vec![];
        if let Some(is_preset) = is_preset {
            conditions.push(format!("is_preset={}",is_preset));
        }
        if let Some(active) = active {
            conditions.push(format!("active={}",active));
        }
        let condition = (!conditions.is_empty()).then_some(conditions.join(" AND "));
        let urls = query_paged_as!(
            WebCrawlerUrlDAO,
            "web_crawler_urls",
            ["id", "url", "active", "is_preset", "web_name", "created_at" as "created_at!: DateTime<Utc>"],
            limit,
            skip_id,
            backwards,
            condition
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(urls)
    }

    pub async fn create_web_crawler_url(&self, name: String, url: String, active: bool, is_preset: bool) -> Result<i64> {
        let res = query!("INSERT INTO web_crawler_urls(web_name, url, active, is_preset) VALUES (?,?,?,?);", name, url, active, is_preset)
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

    pub async fn update_active_web_crawler_url(&self, id: i64, active: bool) -> Result<String> {
        let url: String = query_scalar!("SELECT url FROM web_crawler_urls WHERE id = ?", id)
            .fetch_one(&self.pool)
            .await
            .map_err(|_| anyhow!("The specified web_crawler_urls id does not exist"))?;
        let res = query!(
            "UPDATE web_crawler_urls SET active = $1 WHERE id = $2;",
            id,
            active
        ).execute(&self.pool).await?;

        if res.rows_affected() != 1 {
            return Err(anyhow!(
                "The specified web_crawler_urls id does not exist"
            ));
        }

        Ok(url)
    }
}
