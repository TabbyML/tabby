use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use sqlx::{prelude::FromRow, query};
use tabby_db_macros::query_paged_as;
use serde;
use sqlx;
use juniper::GraphQLEnum;

use crate::DbConn;

#[derive(Debug, sqlx::Type, serde::Serialize, serde::Deserialize,)]
#[sqlx(type_name = "integration_type",rename_all = "lowercase")]
#[derive(Clone, GraphQLEnum)]
pub enum NotionDocumentType {
    Database,
    Page,
    Wiki,
    None,
}

impl From<String> for NotionDocumentType {
    fn from(value: String) -> Self {
        match value.as_str() {
            "database" => NotionDocumentType::Database,
            "page" => NotionDocumentType::Page,
            "wiki" => NotionDocumentType::Wiki,
            // ...
            _ => NotionDocumentType::None,
        }
    }
}

impl Into<String> for NotionDocumentType {
    fn into(self) -> String {
        match self {
            NotionDocumentType::Database => "database".to_string(),
            NotionDocumentType::Page => "page".to_string(),
            NotionDocumentType::Wiki => "wiki".to_string(),
            NotionDocumentType::None => "none".to_string(),
        }
    }
}

#[allow(unused)]
#[derive(FromRow)]
pub struct NotionDocumentDAO {
    pub id: i64,
    pub name: String,
    pub integration_id: String,
    pub integration_type: NotionDocumentType,
    pub access_token: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl DbConn {
    
    pub async fn list_notion_documents(
        &self,
        ids: Option<Vec<i64>>,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<NotionDocumentDAO>> {
        let mut conditions = vec![];

        if let Some(ids) = ids {
            let ids: Vec<String> = ids.iter().map(i64::to_string).collect();
            let ids = ids.join(", ");
            conditions.push(format!("id in ({ids})"));
        }

        let condition = (!conditions.is_empty()).then_some(conditions.join(" AND "));

        let pages = query_paged_as!(
            NotionDocumentDAO,
            "notion_documents",
            ["id", "name", "integration_id","integration_type", "access_token", "created_at" as "created_at!: DateTime<Utc>", "updated_at" as "updated_at!: DateTime<Utc>"],
            limit,
            skip_id,
            backwards,
            condition
        ).fetch_all(&self.pool)
            .await?;

        Ok(pages)
    }

    pub async fn create_notion_document(
        &self,
        name: String,
        integration_id: String,
        integration_type: NotionDocumentType,
        access_token: String,
    ) -> Result<i64> {
        let res = query!(
            "INSERT INTO notion_documents(name, integration_id, integration_type, access_token) VALUES (?,?,?,?);",
            name,
            integration_id,
            integration_type,
            access_token
        )
        .execute(&self.pool)
        .await?;

        Ok(res.last_insert_rowid())
    }


    pub async fn delete_notion_document(&self, id: i64) -> Result<bool> {
        query!("DELETE FROM notion_documents WHERE id = ?;", id)
            .execute(&self.pool)
            .await?;
        Ok(true)
    }
}
