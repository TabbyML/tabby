use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use sqlx::{query, query_scalar, FromRow};

use super::DbConn;

const GITHUB_OAUTH_CREDENTIAL_ROW_ID: i32 = 1;

#[derive(FromRow)]
pub struct GithubOAuthCredentialDAO {
    pub client_id: String,
    pub client_secret: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// db read/write operations for `github_oauth_credential` table
impl DbConn {
    pub async fn update_github_oauth_credential(
        &self,
        client_id: &str,
        client_secret: Option<&str>,
    ) -> Result<()> {
        let client_id = client_id.to_string();
        let mut transaction = self.pool.begin().await?;
        let client_secret = match client_secret {
            Some(secret) => secret.to_string(),
            None => {
                query_scalar!(
                    "SELECT client_secret FROM github_oauth_credential WHERE id = ?",
                    GITHUB_OAUTH_CREDENTIAL_ROW_ID
                )
                .fetch_one(&mut *transaction)
                .await.map_err(|_| anyhow!("Must specify client secret when updating the OAuth credential for the first time"))?
            }
        };
        query!(
            r#"INSERT INTO github_oauth_credential (id, client_id, client_secret)
                                VALUES ($1, $2, $3) ON CONFLICT(id) DO UPDATE
                                SET client_id = $2, client_secret = $3, updated_at = datetime('now')
                                WHERE id = $1"#,
            GITHUB_OAUTH_CREDENTIAL_ROW_ID,
            client_id,
            client_secret
        )
        .execute(&mut *transaction)
        .await?;
        transaction.commit().await?;
        Ok(())
    }

    pub async fn delete_github_oauth_credential(&self) -> Result<()> {
        query!(
            "DELETE FROM github_oauth_credential WHERE id = ?",
            GITHUB_OAUTH_CREDENTIAL_ROW_ID
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn read_github_oauth_credential(&self) -> Result<Option<GithubOAuthCredentialDAO>> {
        let token = sqlx::query_as("SELECT client_id, client_secret, created_at, updated_at FROM github_oauth_credential WHERE id = ?")
            .bind(GITHUB_OAUTH_CREDENTIAL_ROW_ID)
            .fetch_optional(&self.pool).await?;
        Ok(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_update_github_oauth_credential() {
        let conn = DbConn::new_in_memory().await.unwrap();

        // test insert
        conn.update_github_oauth_credential("client_id", Some("client_secret"))
            .await
            .unwrap();
        let res = conn.read_github_oauth_credential().await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id");
        assert_eq!(res.client_secret, "client_secret");

        // test update
        conn.update_github_oauth_credential("client_id", Some("client_secret_2"))
            .await
            .unwrap();
        let res = conn.read_github_oauth_credential().await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id");
        assert_eq!(res.client_secret, "client_secret_2");

        conn.update_github_oauth_credential("client_side", None)
            .await
            .unwrap();
        let res = conn.read_github_oauth_credential().await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id");
        assert_eq!(res.client_secret, "client_secret_2");

        // test delete
        conn.delete_github_oauth_credential().await.unwrap();
        assert!(conn.read_github_oauth_credential().await.unwrap().is_none());

        // test update after delete
        conn.update_github_oauth_credential("client_id_2", Some("client_secret_2"))
            .await
            .unwrap();
        let res = conn.read_github_oauth_credential().await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id_2");
        assert_eq!(res.client_secret, "client_secret_2");
    }
}
