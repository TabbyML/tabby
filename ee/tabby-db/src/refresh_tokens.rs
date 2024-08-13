use anyhow::Result;
use chrono::{DateTime, Utc};
use hash_ids::HashIds;
use lazy_static::lazy_static;
use sqlx::{query, FromRow};
use uuid::Uuid;

use super::{AsSqliteDateTimeString, DbConn};

#[allow(unused)]
#[derive(FromRow)]
pub struct RefreshTokenDAO {
    pub id: i64,
    pub created_at: DateTime<Utc>,

    pub user_id: i64,
    pub token: String,
    pub expires_at: DateTime<Utc>,
}

impl RefreshTokenDAO {
    pub fn is_expired(&self) -> bool {
        let now = Utc::now();
        self.expires_at < now
    }
}

/// db read/write operations for `refresh_tokens` table
impl DbConn {
    pub async fn create_refresh_token(&self, user_id: i64) -> Result<String> {
        let token = generate_refresh_token(0);
        let res = query!(
            r#"INSERT INTO refresh_tokens (user_id, token, expires_at) VALUES (?, ?, datetime('now', '+7 days'))"#,
            user_id,
            token
        ).execute(&self.pool).await?;

        if res.rows_affected() != 1 {
            return Err(anyhow::anyhow!("failed to create refresh token"));
        }

        Ok(token)
    }

    pub async fn renew_refresh_token(&self, id: i64, old: &str) -> Result<String> {
        let new = generate_refresh_token(id);
        let res = query!(
            "UPDATE refresh_tokens SET token = $1, expires_at = datetime('now', '+7 days') WHERE token = $2 AND id = $3",
            new,
            old,
            id
        )
        .execute(&self.pool)
        .await?;

        if res.rows_affected() != 1 {
            return Err(anyhow::anyhow!("failed to replace refresh token"));
        }

        Ok(new)
    }

    pub async fn delete_expired_token(&self) -> Result<i32> {
        let time = Utc::now().as_sqlite_datetime();
        let res = query!(r#"DELETE FROM refresh_tokens WHERE expires_at < ?"#, time)
            .execute(&self.pool)
            .await?;

        Ok(res.rows_affected() as i32)
    }

    pub async fn get_refresh_token(&self, token: &str) -> Result<Option<RefreshTokenDAO>> {
        let token = sqlx::query_as!(
            RefreshTokenDAO,
            r#"SELECT id as "id!", created_at as "created_at!: DateTime<Utc>", expires_at as "expires_at!: DateTime<Utc>", user_id, token FROM refresh_tokens WHERE token = ?"#,
            token
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(token)
    }

    pub async fn delete_tokens_by_user_id(&self, id: i64) -> Result<()> {
        query!("DELETE FROM refresh_tokens WHERE user_id = ?", id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}

lazy_static! {
    static ref HASHER: HashIds = HashIds::builder()
        .with_salt("tabby-refresh-token")
        .with_min_length(6)
        .finish();
}

pub fn generate_refresh_token(id: i64) -> String {
    let uuid = Uuid::new_v4().to_string().replace('-', "");
    let id = HASHER.encode(&[id as u64]);
    format!("{id}{uuid}")
}

#[cfg(test)]
mod tests {

    use super::*;

    #[tokio::test]
    async fn test_create_refresh_token() {
        let conn = DbConn::new_in_memory().await.unwrap();
        let user_id = conn
            .create_user("email@email".into(), None, true, None)
            .await
            .unwrap();
        let token = conn.create_refresh_token(user_id).await.unwrap();

        let dao = conn.get_refresh_token(&token).await.unwrap().unwrap();

        assert_eq!(dao.user_id, 1);
        assert_eq!(dao.token, token);
        assert!(dao.expires_at > Utc::now() + chrono::Duration::days(6));
        assert!(dao.expires_at < Utc::now() + chrono::Duration::days(7));
    }

    #[tokio::test]
    async fn test_replace_refresh_token() {
        let conn = DbConn::new_in_memory().await.unwrap();

        let user_id = conn
            .create_user("email@email".into(), None, true, None)
            .await
            .unwrap();
        let old = conn.create_refresh_token(user_id).await.unwrap();
        let new = conn.renew_refresh_token(1, &old).await.unwrap();

        let token = conn.get_refresh_token(&old).await.unwrap();
        assert!(token.is_none());

        let token = conn.get_refresh_token(&new).await.unwrap().unwrap();
        assert_eq!(token.user_id, 1);
        assert_eq!(token.token, new);
    }

    #[tokio::test]
    async fn test_delete_expired_token() {
        let conn = DbConn::new_in_memory().await.unwrap();
        assert!(conn.delete_expired_token().await.is_ok());
    }
}
