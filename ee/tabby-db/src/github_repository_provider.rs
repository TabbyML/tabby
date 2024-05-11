use anyhow::{anyhow, Result};
use sqlx::{prelude::FromRow, query, query_as};
use tabby_db_macros::query_paged_as;

use crate::{DateTimeUtc, DbConn};

#[derive(FromRow)]
pub struct GithubRepositoryProviderDAO {
    pub id: i64,
    pub display_name: String,
    pub access_token: Option<String>,
    pub synced_at: Option<DateTimeUtc>,
}

#[derive(FromRow)]
pub struct GithubProvidedRepositoryDAO {
    pub id: i64,
    pub vendor_id: String,
    pub github_repository_provider_id: i64,
    pub name: String,
    pub git_url: String,
    pub active: bool,
}

impl DbConn {
    pub async fn create_github_provider(&self, name: String, access_token: String) -> Result<i64> {
        let res = query!(
            "INSERT INTO github_repository_provider (display_name, access_token) VALUES ($1, $2);",
            name,
            access_token
        )
        .execute(&self.pool)
        .await?;
        Ok(res.last_insert_rowid())
    }

    pub async fn get_github_provider(&self, id: i64) -> Result<GithubRepositoryProviderDAO> {
        let provider = query_as!(
            GithubRepositoryProviderDAO,
            r#"SELECT id, display_name, access_token, synced_at AS "synced_at: DateTimeUtc" FROM github_repository_provider WHERE id = ?;"#,
            id
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(provider)
    }

    pub async fn delete_github_provider(&self, id: i64) -> Result<()> {
        let res = query!("DELETE FROM github_repository_provider WHERE id = ?;", id)
            .execute(&self.pool)
            .await?;
        if res.rows_affected() != 1 {
            return Err(anyhow!("No github provider details to delete"));
        }
        Ok(())
    }

    pub async fn update_github_provider(
        &self,
        id: i64,
        display_name: String,
        access_token: Option<String>,
    ) -> Result<()> {
        let access_token = match access_token {
            Some(access_token) => Some(access_token),
            None => self.get_github_provider(id).await?.access_token,
        };

        let res = query!(
            "UPDATE github_repository_provider SET display_name = ?, access_token=? WHERE id = ?;",
            display_name,
            access_token,
            id
        )
        .execute(&self.pool)
        .await?;

        if res.rows_affected() != 1 {
            return Err(anyhow!(
                "The specified Github repository provider does not exist"
            ));
        }

        Ok(())
    }

    pub async fn update_github_provider_sync_status(&self, id: i64, success: bool) -> Result<()> {
        let time = success.then_some(DateTimeUtc::now());
        query!(
            "UPDATE github_repository_provider SET synced_at = ?, access_token = IIF(?, access_token, NULL) WHERE id = ?",
            time,
            success,
            id
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_github_repository_providers(
        &self,
        ids: Vec<i64>,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<GithubRepositoryProviderDAO>> {
        let condition = (!ids.is_empty()).then(|| {
            let ids = ids
                .into_iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("id in ({ids})")
        });
        let providers = query_paged_as!(
            GithubRepositoryProviderDAO,
            "github_repository_provider",
            ["id", "display_name", "access_token", "synced_at" as "synced_at: DateTimeUtc"],
            limit,
            skip_id,
            backwards,
            condition
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(providers)
    }

    pub async fn upsert_github_provided_repository(
        &self,
        github_provider_id: i64,
        vendor_id: String,
        name: String,
        git_url: String,
    ) -> Result<i64> {
        let res = query!(
            "INSERT INTO github_provided_repositories (github_repository_provider_id, vendor_id, name, git_url) VALUES ($1, $2, $3, $4)
                ON CONFLICT(github_repository_provider_id, vendor_id) DO UPDATE SET name = $3, git_url = $4, updated_at = DATETIME('now')",
            github_provider_id,
            vendor_id,
            name,
            git_url
        ).execute(&self.pool).await?;
        Ok(res.last_insert_rowid())
    }

    pub async fn delete_github_provided_repository(&self, id: i64) -> Result<()> {
        let res = query!("DELETE FROM github_provided_repositories WHERE id = ?", id)
            .execute(&self.pool)
            .await?;

        if res.rows_affected() != 1 {
            return Err(anyhow!("Repository not found"));
        }
        Ok(())
    }

    pub async fn delete_outdated_github_repositories(
        &self,
        github_provider_id: i64,
        cutoff_timestamp: DateTimeUtc,
    ) -> Result<usize> {
        let res = query!(
            "DELETE FROM github_provided_repositories WHERE github_repository_provider_id = ? AND updated_at < ?;",
            github_provider_id,
            cutoff_timestamp
        ).execute(&self.pool).await?;
        Ok(res.rows_affected() as usize)
    }

    pub async fn get_github_provided_repository(
        &self,
        id: i64,
    ) -> Result<GithubProvidedRepositoryDAO> {
        let repo = query_as!(
            GithubProvidedRepositoryDAO,
            "SELECT id, vendor_id, name, git_url, active, github_repository_provider_id FROM github_provided_repositories WHERE id = ?",
            id
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(repo)
    }

    pub async fn list_github_provided_repositories(
        &self,
        provider_ids: Vec<i64>,
        active: Option<bool>,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<GithubProvidedRepositoryDAO>> {
        let mut conditions = vec![];

        let provider_ids = provider_ids
            .into_iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        if !provider_ids.is_empty() {
            conditions.push(format!("github_repository_provider_id IN ({provider_ids})"));
        }

        let active_filter = active.map(|active| format!("active = {active}"));
        conditions.extend(active_filter);

        let condition = (!conditions.is_empty()).then(|| conditions.join(" AND "));

        let repos = query_paged_as!(
            GithubProvidedRepositoryDAO,
            "github_provided_repositories",
            [
                "id",
                "vendor_id",
                "name",
                "git_url",
                "active",
                "github_repository_provider_id"
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

    pub async fn update_github_provided_repository_active(
        &self,
        id: i64,
        active: bool,
    ) -> Result<()> {
        let not_active = !active;
        let res = query!(
            "UPDATE github_provided_repositories SET active = ? WHERE id = ? AND active = ?",
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
