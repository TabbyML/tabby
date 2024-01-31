use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::{query, FromRow};

use super::DbConn;

#[allow(unused)]
#[derive(FromRow)]
pub struct RefreshTokenDAO {
    id: u32,
    created_at: DateTime<Utc>,

    pub user_id: i32,
    pub token: String,
    pub expires_at: DateTime<Utc>,
}

impl RefreshTokenDAO {
    pub fn is_expired(&self) -> bool {
        let now = chrono::Utc::now();
        self.expires_at < now
    }
}

/// db read/write operations for `refresh_tokens` table
impl DbConn {
    pub async fn create_refresh_token(&self, user_id: i32, token: &str) -> Result<()> {
        let res = query!(
            r#"INSERT INTO refresh_tokens (user_id, token, expires_at) VALUES (?, ?, datetime('now', '+7 days'))"#,
            user_id,
            token
        ).execute(&self.pool).await?;

        if res.rows_affected() != 1 {
            return Err(anyhow::anyhow!("failed to create refresh token"));
        }

        Ok(())
    }

    pub async fn replace_refresh_token(&self, old: &str, new: &str) -> Result<()> {
        let res = query!(
            "UPDATE refresh_tokens SET token = $1 WHERE token = $2",
            new,
            old
        )
        .execute(&self.pool)
        .await?;

        if res.rows_affected() != 1 {
            return Err(anyhow::anyhow!("failed to replace refresh token"));
        }

        Ok(())
    }

    pub async fn delete_expired_token(&self) -> Result<i32> {
        let time = Utc::now();
        let res = query!(r#"DELETE FROM refresh_tokens WHERE expires_at < ?"#, time)
            .execute(&self.pool)
            .await?;

        Ok(res.rows_affected() as i32)
    }

    pub async fn get_refresh_token(&self, token: &str) -> Result<Option<RefreshTokenDAO>> {
        let token = sqlx::query_as("SELECT * FROM refresh_tokens WHERE token = ?")
            .bind(token)
            .fetch_optional(&self.pool)
            .await?;

        Ok(token)
    }
}

#[cfg(test)]
mod tests {

    use std::ops::Add;

    use super::*;

    #[tokio::test]
    async fn test_create_refresh_token() {
        let conn = DbConn::new_in_memory().await.unwrap();

        conn.create_refresh_token(1, "test").await.unwrap();

        let token = conn.get_refresh_token("test").await.unwrap().unwrap();

        assert_eq!(token.user_id, 1);
        assert_eq!(token.token, "test");
        assert!(token.expires_at > Utc::now().add(chrono::Duration::days(6)));
        assert!(token.expires_at < Utc::now().add(chrono::Duration::days(7)));
    }

    #[tokio::test]
    async fn test_replace_refresh_token() {
        let conn = DbConn::new_in_memory().await.unwrap();

        conn.create_refresh_token(1, "test").await.unwrap();
        conn.replace_refresh_token("test", "test2").await.unwrap();

        let token = conn.get_refresh_token("test").await.unwrap();
        assert!(token.is_none());

        let token = conn.get_refresh_token("test2").await.unwrap().unwrap();
        assert_eq!(token.user_id, 1);
        assert_eq!(token.token, "test2");
    }
}
