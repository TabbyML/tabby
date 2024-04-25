use anyhow::{anyhow, Result};
use sqlx::{prelude::FromRow, query, query_as};
use tabby_db_macros::query_paged_as;

use crate::{DateTimeUtc, DbConn};

#[derive(FromRow)]
pub struct GitlabRepositoryProviderDAO {
    pub id: i64,
    pub display_name: String,
    pub access_token: Option<String>,
}

#[derive(FromRow)]
pub struct GitlabProvidedRepositoryDAO {
    pub id: i64,
    pub vendor_id: String,
    pub gitlab_repository_provider_id: i64,
    pub name: String,
    pub git_url: String,
    pub active: bool,
}

impl DbConn {
    pub async fn create_gitlab_provider(&self, name: String, access_token: String) -> Result<i64> {
        let res = query!(
            "INSERT INTO gitlab_repository_provider (display_name, access_token) VALUES ($1, $2);",
            name,
            access_token
        )
        .execute(&self.pool)
        .await?;
        Ok(res.last_insert_rowid())
    }

    pub async fn get_gitlab_provider(&self, id: i64) -> Result<GitlabRepositoryProviderDAO> {
        let provider = query_as!(
            GitlabRepositoryProviderDAO,
            "SELECT id, display_name, access_token FROM gitlab_repository_provider WHERE id = ?;",
            id
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(provider)
    }

    pub async fn delete_gitlab_provider(&self, id: i64) -> Result<()> {
        let res = query!("DELETE FROM gitlab_repository_provider WHERE id = ?;", id)
            .execute(&self.pool)
            .await?;
        if res.rows_affected() != 1 {
            return Err(anyhow!("No gitlab provider details to delete"));
        }
        Ok(())
    }

    pub async fn reset_gitlab_provider_access_token(&self, id: i64) -> Result<()> {
        let res = query!(
            "UPDATE gitlab_repository_provider SET access_token = NULL WHERE id = ?",
            id
        )
        .execute(&self.pool)
        .await?;

        if res.rows_affected() != 1 {
            return Err(anyhow!(
                "The specified gitlab repository provider does not exist"
            ));
        }

        Ok(())
    }

    pub async fn update_gitlab_provider(
        &self,
        id: i64,
        display_name: String,
        access_token: String,
    ) -> Result<()> {
        let res = query!(
            "UPDATE gitlab_repository_provider SET display_name = ?, access_token=? WHERE id = ?;",
            display_name,
            access_token,
            id
        )
        .execute(&self.pool)
        .await?;

        if res.rows_affected() != 1 {
            return Err(anyhow!(
                "The specified gitlab repository provider does not exist"
            ));
        }

        Ok(())
    }

    pub async fn list_gitlab_repository_providers(
        &self,
        ids: Vec<i64>,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<GitlabRepositoryProviderDAO>> {
        let condition = (!ids.is_empty()).then(|| {
            let ids = ids
                .into_iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("id in ({ids})")
        });
        let providers = query_paged_as!(
            GitlabRepositoryProviderDAO,
            "gitlab_repository_provider",
            ["id", "display_name", "access_token"],
            limit,
            skip_id,
            backwards,
            condition
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(providers)
    }

    pub async fn upsert_gitlab_provided_repository(
        &self,
        gitlab_provider_id: i64,
        vendor_id: String,
        name: String,
        git_url: String,
    ) -> Result<i64> {
        let res = query!(
            "INSERT INTO gitlab_provided_repositories (gitlab_repository_provider_id, vendor_id, name, git_url) VALUES ($1, $2, $3, $4)
                ON CONFLICT(gitlab_repository_provider_id, vendor_id) DO UPDATE SET name = $3, git_url = $4, updated_at = DATETIME('now')",
            gitlab_provider_id,
            vendor_id,
            name,
            git_url
        ).execute(&self.pool).await?;
        Ok(res.last_insert_rowid())
    }

    pub async fn delete_gitlab_provided_repository(&self, id: i64) -> Result<()> {
        let res = query!("DELETE FROM gitlab_provided_repositories WHERE id = ?", id)
            .execute(&self.pool)
            .await?;

        if res.rows_affected() != 1 {
            return Err(anyhow!("Repository not found"));
        }
        Ok(())
    }

    pub async fn delete_outdated_gitlab_repositories(
        &self,
        gitlab_provider_id: i64,
        cutoff_timestamp: DateTimeUtc,
    ) -> Result<()> {
        query!(
            "DELETE FROM gitlab_provided_repositories WHERE gitlab_repository_provider_id = ? AND updated_at < ?;",
            gitlab_provider_id,
            cutoff_timestamp
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn get_gitlab_provided_repository(
        &self,
        id: i64,
    ) -> Result<GitlabProvidedRepositoryDAO> {
        let repo = query_as!(
            GitlabProvidedRepositoryDAO,
            "SELECT id, vendor_id, name, git_url, active, gitlab_repository_provider_id FROM gitlab_provided_repositories WHERE id = ?",
            id
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(repo)
    }

    pub async fn list_gitlab_provided_repositories(
        &self,
        provider_ids: Vec<i64>,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<GitlabProvidedRepositoryDAO>> {
        let provider_ids = provider_ids
            .into_iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let repos = query_paged_as!(
            GitlabProvidedRepositoryDAO,
            "gitlab_provided_repositories",
            [
                "id",
                "vendor_id",
                "name",
                "git_url",
                "active",
                "gitlab_repository_provider_id"
            ],
            limit,
            skip_id,
            backwards,
            (!provider_ids.is_empty())
                .then(|| format!("gitlab_repository_provider_id IN ({provider_ids})"))
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(repos)
    }

    pub async fn update_gitlab_provided_repository_active(
        &self,
        id: i64,
        active: bool,
    ) -> Result<()> {
        let not_active = !active;
        let res = query!(
            "UPDATE gitlab_provided_repositories SET active = ? WHERE id = ? AND active = ?",
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
