use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::{query, FromRow};

use super::DbConn;

const GOOGLE_OAUTH_CREDENTIAL_ROW_ID: i32 = 1;

#[derive(FromRow)]
pub struct GoogleOAuthCredentialDAO {
    pub client_id: String,
    pub client_secret: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// db read/write operations for `google_oauth_credential` table
impl DbConn {
    pub async fn update_google_oauth_credential(
        &self,
        client_id: &str,
        client_secret: &str,
    ) -> Result<()> {
        let client_id = client_id.to_string();
        let client_secret = client_secret.to_string();
        query!(
            r#"INSERT INTO google_oauth_credential (id, client_id, client_secret)
                                VALUES ($1, $2, $3) ON CONFLICT(id) DO UPDATE
                                SET client_id = $2, client_secret = $3, updated_at = datetime('now')
                                WHERE id = $1"#,
            GOOGLE_OAUTH_CREDENTIAL_ROW_ID,
            client_id,
            client_secret,
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn read_google_oauth_credential(&self) -> Result<Option<GoogleOAuthCredentialDAO>> {
        let token = sqlx::query_as(
            r#"SELECT client_id, client_secret, created_at, updated_at FROM google_oauth_credential WHERE id = ?"#,
        ).bind(GOOGLE_OAUTH_CREDENTIAL_ROW_ID).fetch_optional(&self.pool).await?;
        Ok(token)
    }

    pub async fn delete_google_oauth_credential(&self) -> Result<()> {
        query!(
            "DELETE FROM google_oauth_credential WHERE id = ?",
            GOOGLE_OAUTH_CREDENTIAL_ROW_ID
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_update_google_oauth_credential() {
        let conn = DbConn::new_in_memory().await.unwrap();

        // test insert
        conn.update_google_oauth_credential("client_id", "client_secret")
            .await
            .unwrap();
        let res = conn.read_google_oauth_credential().await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id");
        assert_eq!(res.client_secret, "client_secret");

        // test delete
        conn.delete_google_oauth_credential().await.unwrap();
        let res = conn.read_google_oauth_credential().await.unwrap();
        assert!(res.is_none());

        // test insert with redirect_uri
        conn.update_google_oauth_credential("client_id", "client_secret")
            .await
            .unwrap();
        conn.read_google_oauth_credential().await.unwrap().unwrap();

        // test update
        conn.update_google_oauth_credential("client_id_2", "client_secret_2")
            .await
            .unwrap();
        let res = conn.read_google_oauth_credential().await.unwrap().unwrap();
        assert_eq!(res.client_id, "client_id_2");
        assert_eq!(res.client_secret, "client_secret_2");
    }
}
