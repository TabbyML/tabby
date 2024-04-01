use anyhow::{anyhow, Result};
use sqlx::{query, query_as};

use crate::DbConn;

pub struct GithubProviderDAO {
    pub name: String,
    pub github_url: String,
    pub application_id: String,
    pub secret: String,
}

const GITHUB_PROVIDER_ROWID: i64 = 1;

impl DbConn {
    pub async fn update_github_provider(
        &self,
        name: String,
        github_url: String,
        application_id: String,
        secret: String,
    ) -> Result<()> {
        query!("INSERT INTO github_provider (id, name, github_url, application_id, secret) VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT(name) DO UPDATE SET github_url=$3, application_id=$4, secret=$5 WHERE id = $1;",
            GITHUB_PROVIDER_ROWID,
            name,
            github_url,
            application_id,
            secret
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn get_github_provider(&self) -> Result<GithubProviderDAO> {
        let provider = query_as!(
            GithubProviderDAO,
            "SELECT name, github_url, application_id, secret FROM github_provider WHERE id = ?;",
            GITHUB_PROVIDER_ROWID
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(provider)
    }

    pub async fn delete_github_provider(&self) -> Result<()> {
        let res = query!(
            "DELETE FROM github_provider WHERE id = ?;",
            GITHUB_PROVIDER_ROWID
        )
        .execute(&self.pool)
        .await?;
        if res.rows_affected() != 1 {
            return Err(anyhow!("No github provider details to delete"));
        }
        Ok(())
    }
}
