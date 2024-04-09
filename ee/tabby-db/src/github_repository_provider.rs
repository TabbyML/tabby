use anyhow::{anyhow, Result};
use sqlx::{prelude::FromRow, query, query_as};
use tabby_db_macros::query_paged_as;

use crate::{DbConn, SQLXResultExt};

#[derive(FromRow)]
pub struct GithubRepositoryProviderDAO {
    pub id: i64,
    pub display_name: String,
    pub application_id: String,
    pub secret: String,
    pub access_token: Option<String>,
}

#[derive(FromRow)]
pub struct GithubProvidedRepositoryDAO {
    pub id: i64,
    pub github_repository_provider_id: i64,
    pub name: String,
    pub git_url: String,
    pub active: bool,
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

    pub async fn get_github_provider(&self, id: i64) -> Result<GithubRepositoryProviderDAO> {
        let provider = query_as!(
            GithubRepositoryProviderDAO,
            "SELECT id, display_name, application_id, secret, access_token FROM github_repository_provider WHERE id = ?;",
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

    pub async fn update_github_provider_access_token(
        &self,
        id: i64,
        access_token: String,
    ) -> Result<()> {
        let res = query!(
            "UPDATE github_repository_provider SET access_token = ? WHERE id = ?",
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

    pub async fn list_github_repository_providers(
        &self,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<GithubRepositoryProviderDAO>> {
        let providers = query_paged_as!(
            GithubRepositoryProviderDAO,
            "github_repository_provider",
            [
                "id",
                "display_name",
                "application_id",
                "secret",
                "access_token"
            ],
            limit,
            skip_id,
            backwards
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(providers)
    }

    pub async fn create_github_provided_repository(
        &self,
        github_provider_id: i64,
        name: String,
        git_url: String,
    ) -> Result<()> {
        query!("INSERT INTO github_provided_repositories (github_repository_provider_id, name, git_url) VALUES (?, ?, ?)",
            github_provider_id, name, git_url).execute(&self.pool).await?;
        Ok(())
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

    pub async fn list_github_provided_repositories(
        &self,
        provider_id: i64,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<GithubProvidedRepositoryDAO>> {
        let repos = query_paged_as!(
            GithubProvidedRepositoryDAO,
            "github_provided_repositories",
            [
                "id",
                "name",
                "git_url",
                "active",
                "github_repository_provider_id"
            ],
            limit,
            skip_id,
            backwards,
            Some(format!("github_repository_provider_id = {provider_id}"))
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(repos)
    }
}
