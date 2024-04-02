use anyhow::{anyhow, Result};
use sqlx::{query, query_as};

use crate::{DbConn, SQLXResultExt};

pub struct GithubProviderDAO {
    pub display_name: String,
    pub application_id: String,
    pub secret: String,
}

impl DbConn {
    pub async fn update_github_provider(
        &self,
        name: String,
        application_id: String,
        secret: String,
    ) -> Result<()> {
        query!("INSERT INTO github_repository_provider (display_name, application_id, secret) VALUES ($1, $2, $3);",
            name,
            application_id,
            secret
        ).execute(&self.pool).await.unique_error("Provider already exists")?;
        Ok(())
    }

    pub async fn get_github_provider(&self, display_name: String) -> Result<GithubProviderDAO> {
        let provider = query_as!(
            GithubProviderDAO,
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
}
