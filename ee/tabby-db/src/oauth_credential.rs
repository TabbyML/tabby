use crate::{DateTimeUtc, DbConn};
use anyhow::{anyhow, Result};
use sqlx::{prelude::FromRow, query, query_scalar};

#[derive(FromRow)]
pub struct OAuthCredentialDAO {
    pub provider: String,
    pub client_id: String,
    pub client_secret: String,
    pub created_at: DateTimeUtc,
    pub updated_at: DateTimeUtc,
}

const OAUTH_CREDENTIAL_ROW_ID: i32 = 1;

impl DbConn {
    pub async fn update_oauth_credential(
        &self,
        provider: &str,
        client_id: &str,
        client_secret: Option<&str>,
    ) -> Result<()> {
        let client_id = client_id.to_string();
        let mut transaction = self.pool.begin().await?;
        let client_secret = match client_secret {
            Some(secret) => secret.to_string(),
            None => {
                query_scalar!(
                    "SELECT client_secret FROM oauth_credential WHERE id = ? AND provider = ?",
                    OAUTH_CREDENTIAL_ROW_ID,
                    provider
                )
                .fetch_one(&mut *transaction)
                .await.map_err(|_| anyhow!("Must specify client secret when updating the OAuth credential for the first time"))?
            }
        };
        query!(
            r#"INSERT INTO oauth_credential (id, provider, client_id, client_secret)
                                VALUES ($1, $2, $3, $4) ON CONFLICT(id) DO UPDATE
                                SET client_id = $3, client_secret = $4, updated_at = datetime('now')
                                WHERE id = $1 AND provider = $2"#,
            OAUTH_CREDENTIAL_ROW_ID,
            provider,
            client_id,
            client_secret,
        )
        .execute(&mut *transaction)
        .await?;
        transaction.commit().await?;
        Ok(())
    }

    pub async fn delete_oauth_credential(&self, provider: &str) -> Result<()> {
        query!(
            "DELETE FROM oauth_credential WHERE id = ? AND provider = ?",
            OAUTH_CREDENTIAL_ROW_ID,
            provider
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn read_oauth_credential(
        &self,
        provider: &str,
    ) -> Result<Option<OAuthCredentialDAO>> {
        let token = sqlx::query_as(
            "SELECT provider, client_id, client_secret, created_at, updated_at FROM oauth_credential WHERE id = ? AND provider = ?",
        ).bind(OAUTH_CREDENTIAL_ROW_ID).bind(provider)
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
        conn.update_oauth_credential("github", "client_id", Some("client_secret"))
            .await
            .unwrap();
        let res = conn.read_oauth_credential("github").await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id");
        assert_eq!(res.client_secret, "client_secret");

        // test update
        conn.update_oauth_credential("github", "client_id", Some("client_secret_2"))
            .await
            .unwrap();
        let res = conn.read_oauth_credential("github").await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id");
        assert_eq!(res.client_secret, "client_secret_2");

        conn.update_oauth_credential("github", "client_id", None)
            .await
            .unwrap();
        let res = conn.read_oauth_credential("github").await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id");
        assert_eq!(res.client_secret, "client_secret_2");

        // test delete
        conn.delete_oauth_credential("github").await.unwrap();
        assert!(conn
            .read_oauth_credential("github")
            .await
            .unwrap()
            .is_none());

        // test update after delete
        conn.update_oauth_credential("github", "client_id_2", Some("client_secret_2"))
            .await
            .unwrap();
        let res = conn.read_oauth_credential("github").await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id_2");
        assert_eq!(res.client_secret, "client_secret_2");
    }

    #[tokio::test]
    async fn test_update_google_oauth_credential() {
        let conn = DbConn::new_in_memory().await.unwrap();

        // test insert
        conn.update_oauth_credential("google", "client_id", Some("client_secret"))
            .await
            .unwrap();
        let res = conn.read_oauth_credential("google").await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id");
        assert_eq!(res.client_secret, "client_secret");

        // test delete
        conn.delete_oauth_credential("google").await.unwrap();
        let res = conn.read_oauth_credential("google").await.unwrap();
        assert!(res.is_none());

        // test insert with redirect_uri
        conn.update_oauth_credential("google", "client_id", Some("client_secret"))
            .await
            .unwrap();
        conn.read_oauth_credential("google").await.unwrap().unwrap();

        conn.update_oauth_credential("google", "client_id", None)
            .await
            .unwrap();
        let res = conn.read_oauth_credential("google").await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id");
        assert_eq!(res.client_secret, "client_secret");

        // test update
        conn.update_oauth_credential("google", "client_id_2", Some("client_secret_2"))
            .await
            .unwrap();
        let res = conn.read_oauth_credential("google").await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id_2");
        assert_eq!(res.client_secret, "client_secret_2");
    }
}
