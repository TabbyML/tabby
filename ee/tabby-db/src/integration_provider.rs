use anyhow::{anyhow, Result};
use sqlx::{prelude::FromRow, query, query_as};
use tabby_db_macros::query_paged_as;

use crate::{DateTimeUtc, DbConn};

#[derive(FromRow)]
pub struct IntegrationAccessTokenDAO {
    pub id: i64,
    pub kind: String,
    pub display_name: String,
    pub access_token: Option<String>,
    pub created_at: DateTimeUtc,
    pub updated_at: DateTimeUtc,
}

#[derive(FromRow)]
pub struct ProvidedRepositoryDAO {
    pub id: i64,
    pub vendor_id: String,
    pub access_token_provider_id: i64,
    pub name: String,
    pub git_url: String,
    pub active: bool,
    pub created_at: DateTimeUtc,
    pub updated_at: DateTimeUtc,
}

impl DbConn {
    pub async fn create_integration_access_token(
        &self,
        kind: &str,
        name: &str,
        access_token: &str,
    ) -> Result<i64> {
        let res = query!(
            "INSERT INTO integration_access_tokens(kind, display_name, access_token) VALUES (?, ?, ?);",
            kind,
            name,
            access_token
        )
        .execute(&self.pool)
        .await?;
        Ok(res.last_insert_rowid())
    }

    pub async fn get_integration_access_token(&self, id: i64) -> Result<IntegrationAccessTokenDAO> {
        let provider = query_as!(
            IntegrationAccessTokenDAO,
            r#"SELECT
                id,
                kind,
                display_name,
                access_token,
                created_at AS "created_at: DateTimeUtc",
                updated_at AS "updated_at: DateTimeUtc"
            FROM integration_access_tokens WHERE id = ?;"#,
            id
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(provider)
    }

    pub async fn delete_integration_access_token(&self, id: i64) -> Result<()> {
        let res = query!("DELETE FROM integration_access_tokens WHERE id = ?;", id)
            .execute(&self.pool)
            .await?;
        if res.rows_affected() != 1 {
            return Err(anyhow!("No integration access token to delete"));
        }
        Ok(())
    }

    pub async fn update_integration_access_token(
        &self,
        id: i64,
        display_name: String,
        access_token: String,
    ) -> Result<()> {
        let res = query!(
            "UPDATE integration_access_tokens SET display_name = ?, access_token=? WHERE id = ?;",
            display_name,
            access_token,
            id
        )
        .execute(&self.pool)
        .await?;

        if res.rows_affected() != 1 {
            return Err(anyhow!(
                "The specified integration access token does not exist"
            ));
        }

        Ok(())
    }

    pub async fn update_integration_access_token_sync_status(
        &self,
        id: i64,
        success: bool,
    ) -> Result<()> {
        let time = success.then_some(DateTimeUtc::now());
        query!(
            "UPDATE integration_access_tokens SET updated_at = ?, access_token = IIF(?, access_token, NULL) WHERE id = ?",
            time,
            success,
            id
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_integration_access_tokens(
        &self,
        ids: Vec<i64>,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<IntegrationAccessTokenDAO>> {
        let condition = (!ids.is_empty()).then(|| {
            let ids = ids
                .into_iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("id in ({ids})")
        });
        let providers = query_paged_as!(
            IntegrationAccessTokenDAO,
            "integration_access_tokens",
            [
                "id",
                "kind",
                "display_name",
                "access_token",
                "created_at" as "created_at: DateTimeUtc",
                "updated_at" as "updated_at: DateTimeUtc"
            ],
            limit,
            skip_id,
            backwards,
            condition
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(providers)
    }

    pub async fn upsert_provided_repository(
        &self,
        integration_access_token_id: i64,
        vendor_id: String,
        name: String,
        git_url: String,
    ) -> Result<i64> {
        let res = query!(
            "INSERT INTO provided_repositories (access_token_provider_id, vendor_id, name, git_url) VALUES ($1, $2, $3, $4)
                ON CONFLICT(access_token_provider_id, vendor_id) DO UPDATE SET name = $3, git_url = $4, updated_at = DATETIME('now')",
            integration_access_token_id,
            vendor_id,
            name,
            git_url
        ).execute(&self.pool).await?;
        Ok(res.last_insert_rowid())
    }

    pub async fn delete_outdated_provided_repositories(
        &self,
        integration_access_token_id: i64,
        cutoff_timestamp: DateTimeUtc,
    ) -> Result<usize> {
        let res = query!(
            "DELETE FROM provided_repositories WHERE access_token_provider_id = ? AND updated_at < ?;",
            integration_access_token_id,
            cutoff_timestamp
        ).execute(&self.pool).await?;
        Ok(res.rows_affected() as usize)
    }

    pub async fn get_provided_repository(&self, id: i64) -> Result<ProvidedRepositoryDAO> {
        let repo = query_as!(
            ProvidedRepositoryDAO,
            "SELECT id, vendor_id, name, git_url, active, access_token_provider_id, created_at, updated_at FROM provided_repositories WHERE id = ?",
            id
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(repo)
    }

    pub async fn list_provided_repositories(
        &self,
        provider_ids: Vec<i64>,
        active: Option<bool>,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<ProvidedRepositoryDAO>> {
        let mut conditions = vec![];

        let provider_ids = provider_ids
            .into_iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        if !provider_ids.is_empty() {
            conditions.push(format!("access_token_provider_id IN ({provider_ids})"));
        }

        let active_filter = active.map(|active| format!("active = {active}"));
        conditions.extend(active_filter);

        let condition = (!conditions.is_empty()).then(|| conditions.join(" AND "));

        let repos = query_paged_as!(
            ProvidedRepositoryDAO,
            "provided_repositories",
            [
                "id",
                "vendor_id",
                "name",
                "git_url",
                "active",
                "access_token_provider_id",
                "created_at" as "created_at: DateTimeUtc",
                "updated_at" as "updated_at: DateTimeUtc"
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
}
