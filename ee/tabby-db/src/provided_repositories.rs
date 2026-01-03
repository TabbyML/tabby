use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use sqlx::{prelude::FromRow, query, query_as};
use tabby_db_macros::query_paged_as;

use crate::{AsSqliteDateTimeString, DbConn};

#[derive(FromRow)]
pub struct ProvidedRepositoryDAO {
    pub id: i64,
    pub vendor_id: String,
    pub integration_id: i64,
    pub name: String,
    pub git_url: String,
    pub refs: Option<String>,
    pub active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl DbConn {
    pub async fn upsert_provided_repository(
        &self,
        integration_id: i64,
        vendor_id: String,
        name: String,
        git_url: String,
        refs: Option<Vec<String>>,
    ) -> Result<i64> {
        let refs = refs
            .map(|mut r| {
                if r.is_empty() {
                    r.push("main".to_string());
                }
                r
            })
            .map(|r| serde_json::to_string(&r))
            .transpose()?;
        let res = query!(
            "INSERT INTO provided_repositories (integration_id, vendor_id, name, git_url, refs) VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT(integration_id, vendor_id) DO UPDATE SET name = $3, git_url = $4, refs = COALESCE($5, refs), updated_at = DATETIME('now')",
            integration_id,
            vendor_id,
            name,
            git_url,
            refs
        ).execute(&self.pool).await?;
        Ok(res.last_insert_rowid())
    }

    pub async fn delete_outdated_provided_repositories(
        &self,
        integration_id: i64,
        cutoff_timestamp: DateTime<Utc>,
    ) -> Result<usize> {
        let t = cutoff_timestamp.as_sqlite_datetime();
        let res = query!(
            "DELETE FROM provided_repositories WHERE integration_id = ? AND updated_at < ?;",
            integration_id,
            t,
        )
        .execute(&self.pool)
        .await?;
        Ok(res.rows_affected() as usize)
    }

    pub async fn get_provided_repository(&self, id: i64) -> Result<ProvidedRepositoryDAO> {
        let repo = query_as!(
            ProvidedRepositoryDAO,
            r#"SELECT id, vendor_id, name, git_url, refs, active, integration_id, created_at as "created_at!: DateTime<Utc>", updated_at as "updated_at!: DateTime<Utc>" FROM provided_repositories WHERE id = ?"#,
            id
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(repo)
    }

    pub async fn list_provided_repositories(
        &self,
        integration_ids: Vec<i64>,
        kind: Option<String>,
        active: Option<bool>,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<ProvidedRepositoryDAO>> {
        let mut conditions = vec![];

        let integration_ids = integration_ids
            .into_iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        if !integration_ids.is_empty() {
            conditions.push(format!("integration_id IN ({integration_ids})"));
        }

        let active_filter = active.map(|active| format!("active = {active}"));
        conditions.extend(active_filter);

        let kind_filter = kind.map(|kind| format!("kind = '{kind}'"));
        conditions.extend(kind_filter);

        let condition = (!conditions.is_empty()).then(|| conditions.join(" AND "));

        let repos = query_paged_as!(
            ProvidedRepositoryDAO,
            "provided_repositories JOIN integrations ON integration_id = integrations.id",
            [
                "id",
                "vendor_id",
                "name",
                "git_url",
                "refs",
                "active",
                "integration_id",
                "created_at" as "created_at!: DateTime<Utc>",
                "updated_at" as "updated_at!: DateTime<Utc>"
            ],
            limit,
            skip_id,
            backwards,
            condition
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(repos)
    }

    pub async fn update_provided_repository_active(&self, id: i64, active: bool) -> Result<()> {
        let not_active = !active;
        let res = query!(
            "UPDATE provided_repositories SET active = ? WHERE id = ? AND active = ?",
            active,
            id,
            not_active
        )
        .execute(&self.pool)
        .await?;

        if res.rows_affected() != 1 {
            return Err(anyhow!("Repository active status was not changed"));
        }

        Ok(())
    }

    pub async fn update_provided_repository_refs(
        &self,
        id: i64,
        mut refs: Vec<String>,
    ) -> Result<()> {
        if refs.is_empty() {
            refs.push("main".into());
        }
        let refs = serde_json::to_string(&refs)?;
        let res = query!(
            "UPDATE provided_repositories SET refs = ? WHERE id = ?",
            refs,
            id
        )
        .execute(&self.pool)
        .await?;

        if res.rows_affected() != 1 {
            return Err(anyhow!("Repository not found"));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::DbConn;

    #[tokio::test]
    async fn test_list_provided_repositories() {
        let db = DbConn::new_in_memory().await.unwrap();
        // Ensure query does not break on the join
        db.list_provided_repositories(vec![], Some("github".into()), None, None, None, false)
            .await
            .unwrap();
    }
}
