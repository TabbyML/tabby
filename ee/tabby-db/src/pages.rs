use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::{query, FromRow};
use tabby_db_macros::query_paged_as;

use crate::DbConn;

#[derive(FromRow)]
pub struct PageDAO {
    pub id: i64,
    pub author_id: i64,
    pub title: Option<String>,
    pub summary: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(sqlx::FromRow)]
pub struct PageSectionDAO {
    pub id: i64,
    pub page_id: i64,

    pub position: i64,
    pub title: String,
    pub content: String,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl DbConn {
    pub async fn create_page(&self, author_id: i64) -> Result<i64> {
        let res = query!("INSERT INTO pages(author_id) VALUES (?)", author_id,)
            .execute(&self.pool)
            .await?;

        Ok(res.last_insert_rowid())
    }

    pub async fn create_page_section(
        &self,
        page_id: i64,
        position: i64,
        title: &str,
        content: &str,
    ) -> Result<i64> {
        let res = query!(
            "INSERT INTO page_sections(page_id, position, title, content) VALUES (?, ?, ?, ?)",
            page_id,
            position,
            title,
            content
        )
        .execute(&self.pool)
        .await?;

        Ok(res.last_insert_rowid())
    }

    pub async fn update_page_title(&self, page_id: i64, title: &str) -> Result<()> {
        query!(
            "UPDATE pages SET title = ?, updated_at = DATETIME('now') WHERE id = ?",
            title,
            page_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn update_page_summary(&self, page_id: i64, summary: &str) -> Result<()> {
        query!(
            "UPDATE pages SET summary = ?, updated_at = DATETIME('now') WHERE id = ?",
            summary,
            page_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn list_pages(
        &self,
        ids: Option<&[i64]>,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<PageDAO>> {
        let condition = match ids {
            Some(ids) => format!(
                "id IN ({})",
                ids.iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<String>>()
                    .join(",")
            ),
            None => "1 = 1".to_string(),
        };

        let pages = query_paged_as!(
            PageDAO,
            "pages",
            [
                "id",
                "author_id",
                "title",
                "summary",
                "created_at" as "created_at: DateTime<Utc>",
                "updated_at" as "updated_at: DateTime<Utc>"
            ],
            limit,
            skip_id,
            backwards,
            Some(condition)
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(pages)
    }

    pub async fn delete_page(&self, id: i64) -> Result<()> {
        query!("DELETE FROM pages WHERE id = ?", id,)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    pub async fn list_page_sections(
        &self,
        page_id: i64,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<PageSectionDAO>> {
        let condition = format!("page_id = {}", page_id);
        let sections = query_paged_as!(
            PageSectionDAO,
            "page_sections",
            [
                "id",
                "page_id",
                "position",
                "title",
                "content",
                "created_at" as "created_at: DateTime<Utc>",
                "updated_at" as "updated_at: DateTime<Utc>"
            ],
            limit,
            skip_id,
            backwards,
            Some(condition)
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(sections)
    }
}
