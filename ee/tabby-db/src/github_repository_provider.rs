use anyhow::{anyhow, Result};
use sqlx::{query, query_as};

use crate::{DbConn, SQLXResultExt};

pub struct GithubRepositoryProviderDAO {
    pub display_name: String,
    pub application_id: String,
    pub secret: String,
}

pub struct GithubProvidedRepositoryDAO {
    pub provider_id: i64,
    pub candidate_id: String,
    pub name: String,
    pub git_url: String,
}

impl DbConn {
    pub async fn create_github_provider(
        &self,
        name: String,
        application_id: String,
        secret: String,
    ) -> Result<i64> {
        let res = query!("INSERT INTO github_repository_provider (display_name, application_id, secret) VALUES ($1, $2, $3);",
            name,
            application_id,
            secret
        ).execute(&self.pool).await.unique_error("GitHub Application ID already exists")?;
        Ok(res.last_insert_rowid())
    }

    pub async fn get_github_provider(
        &self,
        display_name: String,
    ) -> Result<GithubRepositoryProviderDAO> {
        let provider = query_as!(
            GithubRepositoryProviderDAO,
            "SELECT display_name, application_id, secret FROM github_repository_provider WHERE display_name = ?;",
            display_name
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(provider)
    }

    pub async fn delete_github_provider(&self, display_name: String) -> Result<()> {
        let res = query!(
            "DELETE FROM github_repository_provider WHERE display_name = ?;",
            display_name
        )
        .execute(&self.pool)
        .await?;
        if res.rows_affected() != 1 {
            return Err(anyhow!("No github provider details to delete"));
        }
        Ok(())
    }

    pub async fn create_github_provided_repository(
        &self,
        provider_id: i64,
        candidate_id: String,
        name: String,
        git_url: String,
    ) -> Result<i64> {
        let res = query!(
            "INSERT INTO github_provided_repositories(provider_id, candidate_id, name, git_url) VALUES (?, ?, ?, ?)",
            provider_id, candidate_id, name, git_url
        ).execute(&self.pool).await?;
        Ok(res.last_insert_rowid())
    }

    pub async fn delete_github_provided_repository(
        &self,
        provider_id: i64,
        candidate_id: String,
    ) -> Result<()> {
        let res = query!(
            "DELETE FROM github_provided_repositories WHERE provider_id = ? AND candidate_id = ?",
            provider_id,
            candidate_id
        )
        .execute(&self.pool)
        .await?;
        if res.rows_affected() == 0 {
            return Err(anyhow!("Specified repository does not exist"));
        }
        Ok(())
    }
}
