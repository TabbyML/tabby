use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::{query, FromRow};
use tabby_db_macros::query_paged_as;

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
    pub status: IngestedDocumentStatusDAO,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(sqlx::Type)]
#[sqlx(rename_all = "lowercase")]
pub enum IngestedDocumentStatusDAO {
    Pending,
    Indexed,
    Failed,
}

/// db read/write operations for `job_runs` table
impl DbConn {
    pub async fn list_ingested_documents(
        &self,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<IngestedDocumentDAO>> {
        let docs = query_paged_as!(
            IngestedDocumentDAO,
            "ingested_documents",
            [
                "id",
                "source",
                "doc_id",
                "expired_at",
                "link",
                "title",
                "body",
                "status" as "status: IngestedDocumentStatusDAO",
                "created_at" as "created_at: DateTime<Utc>",
                "updated_at" as "updated_at: DateTime<Utc>"
            ],
            limit,
            skip_id,
            backwards
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(docs)
    }

    pub async fn list_ingested_document_sources(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Vec<String>> {
        let mut query =
            String::from("SELECT DISTINCT source FROM ingested_documents ORDER BY source");
        if limit.is_some() {
            query.push_str(" LIMIT ?");
        }
        if offset.is_some() {
            query.push_str(" OFFSET ?");
        }

        let mut q = sqlx::query_scalar::<_, String>(&query);
        if let Some(l) = limit {
            q = q.bind(l as i64);
        }
        if let Some(o) = offset {
            q = q.bind(o as i64);
        }

        Ok(q.fetch_all(&self.pool).await?)
    }

    pub async fn upsert_ingested_document(
        &self,
        source: &str,
        doc_id: &str,
        expired_at: i64,
        link: Option<String>,
        title: &str,
        body: &str,
    ) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO ingested_documents (
              source, doc_id, expired_at, link, title, body, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source, doc_id) DO UPDATE SET
              expired_at = excluded.expired_at,
              link = excluded.link,
              title = excluded.title,
              body = excluded.body,
              status = excluded.status
            "#,
            source,
            doc_id,
            expired_at,
            link,
            title,
            body,
            IngestedDocumentStatusDAO::Pending,
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn count_pending_ingested_documents(&self) -> Result<i32> {
        let count = query!(
            r#"
            SELECT COUNT(1) as count
            FROM ingested_documents
            WHERE status = ?
            "#,
            IngestedDocumentStatusDAO::Pending,
        )
        .fetch_one(&self.pool)
        .await?
        .count;

        Ok(count)
    }

    pub async fn mark_ingested_document_indexed(&self, source: &str, doc_id: &str) -> Result<()> {
        sqlx::query!(
            r#"
            UPDATE ingested_documents
            SET status = ?
            WHERE source = ? AND doc_id = ?
            "#,
            IngestedDocumentStatusDAO::Indexed,
            source,
            doc_id,
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}
