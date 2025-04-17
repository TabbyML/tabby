use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::{query, FromRow};

use super::DbConn;

#[derive(FromRow)]
pub struct IngestedDocumentDAO {
    pub id: i64,
    pub source: String,
    pub doc_id: String,
    pub expired_at: i64,
    pub link: Option<String>,
    pub title: String,
    pub body: String,
    pub index_status: IngestedDocumentStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(sqlx::Type)]
#[sqlx(rename_all = "lowercase")]
pub enum IngestedDocumentStatus {
    Pending,
    Indexed,
    Failed,
}

/// db read/write operations for `job_runs` table
impl DbConn {
    pub async fn insert_ingested_document(
        &self,
        source: &str,
        doc_id: &str,
        expired_at: i64,
        link: Option<String>,
        title: &str,
        body: &str,
    ) -> Result<()> {
        query!(
            r#"
            INSERT INTO ingested_documents (
              source, doc_id, expired_at, link, title, body, index_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
            source,
            doc_id,
            expired_at,
            link,
            title,
            body,
            IngestedDocumentStatus::Pending,
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}
