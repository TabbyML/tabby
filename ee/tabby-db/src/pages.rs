use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::{query, query_as, FromRow};
use tabby_db_macros::query_paged_as;

use crate::DbConn;

#[derive(FromRow)]
pub struct PageDAO {
    pub id: i64,
    pub author_id: i64,
    pub title: Option<String>,
    pub content: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(sqlx::FromRow)]
pub struct PageSectionDAO {
    pub id: i64,
    pub page_id: i64,

    pub title: String,
    pub content: Option<String>,
    pub position: i64,

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

    pub async fn append_page_content(&self, page_id: i64, content: &str) -> Result<()> {
        query!(
            "UPDATE pages SET content = COALESCE(content, '') || ?, updated_at = DATETIME('now') WHERE id = ?",
            content,
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

        Ok(pages)
    }

    pub async fn delete_page(&self, id: i64) -> Result<()> {
        query!("DELETE FROM pages WHERE id = ?", id,)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    pub async fn get_page(&self, id: i64) -> Result<Option<PageDAO>> {
        let page = query_as!(
            PageDAO,
            r#"SELECT
                id,
                author_id,
                title,
                content,
                created_at as "created_at: DateTime<Utc>",
                updated_at  as "updated_at: DateTime<Utc>"
            FROM pages
            WHERE id = ?"#,
            id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(page)
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
                "title",
                "content",
                "position",
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

    pub async fn get_page_section(&self, id: i64) -> Result<Option<PageSectionDAO>> {
        let section = query_as!(
            PageSectionDAO,
            r#"SELECT
                id,
                page_id,
                title,
                position,
                content,
                created_at as "created_at: DateTime<Utc>",
                updated_at  as "updated_at: DateTime<Utc>"
            FROM page_sections
            WHERE id = ?"#,
            id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(section)
    }

    pub async fn create_page_section(
        &self,
        page_id: i64,
        title: &str,
        position: i32,
    ) -> Result<i64> {
        let res = query!(
            "INSERT INTO page_sections(page_id, title, position) VALUES (?, ?, ?)",
            page_id,
            title,
            position,
        )
        .execute(&self.pool)
        .await?;

        Ok(res.last_insert_rowid())
    }

    pub async fn append_page_section_content(&self, id: i64, content: &str) -> Result<()> {
        query!(
            "UPDATE page_sections SET content = COALESCE(content, '') || ?, updated_at = DATETIME('now') WHERE id = ?",
            content,
            id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn delete_page_section(&self, id: i64) -> Result<()> {
        query!("DELETE FROM page_sections WHERE id = ?", id,)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    pub async fn update_page_section_position(
        &self,
        page_id: i64,
        id: i64,
        change: i64,
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;

        let current_position = query!("SELECT position FROM page_sections WHERE id = ?", id)
            .fetch_one(&mut *tx)
            .await?
            .position;
        if current_position == 0 && change < 0 {
            // Already at the top
            tx.commit().await?;
            return Ok(());
        }

        let new_position = current_position + change;
        if current_position == new_position {
            // No change needed
            tx.commit().await?;
            return Ok(());
        }

        let max_position = query!(
            "SELECT MAX(position) as max_position FROM page_sections WHERE page_id = ?",
            page_id
        )
        .fetch_one(&mut *tx)
        .await?
        .max_position
        .ok_or_else(|| anyhow::anyhow!("Failed to get max position"))?;

        if current_position == max_position && change > 0 {
            // Already at the bottom
            tx.commit().await?;
            return Ok(());
        }

        query!("UPDATE page_sections SET position = ? WHERE id = ?", -1, id)
            .execute(&mut *tx)
            .await?;

        let (min, max) = if current_position < new_position {
            (current_position, new_position)
        } else {
            (new_position, current_position)
        };
        let sections = query!(
            "SELECT id, position FROM page_sections WHERE page_id = ? AND position BETWEEN ? AND ? ORDER BY position",
            page_id,
            min,
            max,
        ).fetch_all(&mut *tx).await?;

        if current_position < new_position {
            for section in sections {
                let move_position = section.position - 1;
                query!(
                    "UPDATE page_sections SET position = ? WHERE id = ?",
                    move_position,
                    section.id
                )
                .execute(&mut *tx)
                .await?;
            }
        } else {
            for section in sections.iter().rev() {
                let move_position = section.position + 1;
                query!(
                    "UPDATE page_sections SET position = ? WHERE id = ?",
                    move_position,
                    section.id
                )
                .execute(&mut *tx)
                .await?;
            }
        }

        query!(
            "UPDATE page_sections SET position = ? WHERE id = ?",
            new_position,
            id
        )
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;
        Ok(())
    }
}
