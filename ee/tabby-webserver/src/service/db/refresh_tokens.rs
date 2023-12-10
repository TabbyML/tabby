use anyhow::Result;
use chrono::{DateTime, Utc};
use rusqlite::{params, OptionalExtension, Row};

use super::DbConn;

#[allow(unused)]
pub struct RefreshToken {
    id: u32,
    created_at: DateTime<Utc>,

    pub user_id: i32,
    pub token: String,
    pub expires_at: DateTime<Utc>,
}

impl RefreshToken {
    fn select(clause: &str) -> String {
        r#"SELECT id, user_id, token, expires_at, created_at FROM refresh_tokens WHERE "#.to_owned()
            + clause
    }

    fn from_row(row: &Row<'_>) -> std::result::Result<RefreshToken, rusqlite::Error> {
        Ok(RefreshToken {
            id: row.get(0)?,
            user_id: row.get(1)?,
            token: row.get(2)?,
            expires_at: row.get(3)?,
            created_at: row.get(4)?,
        })
    }

    pub fn is_expired(&self) -> bool {
        let now = chrono::Utc::now();
        self.expires_at < now
    }
}

/// db read/write operations for `refresh_tokens` table
impl DbConn {
    pub async fn create_refresh_token(&self, user_id: i32, token: &str) -> Result<()> {
        let token = token.to_string();
        let res = self
            .conn
            .call(move |c| {
                Ok(c.execute(
                    r#"INSERT INTO refresh_tokens (user_id, token, expires_at) VALUES (?, ?, datetime('now', '+7 days'))"#,
                    params![user_id, token],
                ))
            })
            .await?;
        if res != Ok(1) {
            return Err(anyhow::anyhow!("failed to create refresh token"));
        }

        Ok(())
    }

    pub async fn replace_refresh_token(&self, old: &str, new: &str) -> Result<()> {
        let old = old.to_string();
        let new = new.to_string();
        let res = self
            .conn
            .call(move |c| {
                Ok(c.execute(
                    r#"UPDATE refresh_tokens SET token = ? WHERE token = ?"#,
                    params![new, old],
                ))
            })
            .await?;
        if res != Ok(1) {
            return Err(anyhow::anyhow!("failed to replace refresh token"));
        }

        Ok(())
    }

    pub async fn delete_expired_token(&self) -> Result<i32> {
        let res = self
            .conn
            .call(move |c| {
                Ok(c.execute(
                    r#"DELETE FROM refresh_tokens WHERE expires_at < ?"#,
                    params![Utc::now()],
                ))
            })
            .await?;

        Ok(res? as i32)
    }

    pub async fn get_refresh_token(&self, token: &str) -> Result<Option<RefreshToken>> {
        let token = token.to_string();
        let token = self
            .conn
            .call(move |c| {
                Ok(c.query_row(
                    RefreshToken::select("token = ?").as_str(),
                    params![token],
                    RefreshToken::from_row,
                )
                .optional())
            })
            .await?;

        Ok(token?)
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
