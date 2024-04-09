use anyhow::{anyhow, Result};
use sqlx::{prelude::FromRow, query, query_as};

use crate::{DbConn, SQLXResultExt};

pub struct GithubRepositoryProviderDAO {
    pub display_name: String,
    pub application_id: String,
    pub secret: String,
}

#[derive(FromRow)]
pub struct GithubProvidedRepositoryDAO {
    pub github_repository_provider_id: i64,
    pub vendor_id: String,
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

    pub async fn get_github_provider(&self, id: i64) -> Result<GithubRepositoryProviderDAO> {
        let provider = query_as!(
            GithubRepositoryProviderDAO,
            "SELECT display_name, application_id, secret FROM github_repository_provider WHERE id = ?;",
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

    pub async fn update_github_provider_token(&self, id: i64, access_token: String) -> Result<()> {
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
}
