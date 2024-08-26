use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use sqlx::{prelude::FromRow, query};
use tabby_db_macros::query_paged_as;

use crate::DbConn;

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
    pub async fn list_web_documents(
        &self,
        names: Option<Vec<String>>,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
        is_preset: bool,
    ) -> Result<Vec<WebDocumentDAO>> {
        let mut condition = format!("is_preset={}", is_preset);
        if let Some(names) = names {
            let names = names
                .into_iter()
                .map(|s| "\"".to_string() + s.as_str() + "\"")
                .collect::<Vec<_>>()
                .join(",");
            condition += &format!(" AND name in ({names})");
        }
        let condition = Some(condition);

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

    pub async fn deactivate_preset_web_document(&self, name: String) -> Result<()> {
        let res = query!("DELETE FROM web_documents WHERE name = ?;", name)
            .execute(&self.pool)
            .await?;
        if res.rows_affected() != 1 {
            return Err(anyhow!("No preset web document to deactivate"));
        }
        Ok(())
    }

    pub async fn delete_web_document(&self, id: i64) -> Result<()> {
        query!("DELETE FROM web_documents WHERE id = ?;", id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}
