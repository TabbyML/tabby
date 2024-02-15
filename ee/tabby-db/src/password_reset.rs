use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use sqlx::{prelude::FromRow, query};
use uuid::Uuid;

use crate::DbConn;

#[derive(FromRow)]
pub struct PasswordResetDAO {
    pub user_id: i32,
    pub code: String,
    pub created_at: DateTime<Utc>,
}

impl DbConn {
    pub async fn create_password_reset(&self, user_id: i32) -> Result<String> {
        let code = Uuid::new_v4().to_string();
        let time = Utc::now();
        query!(
            "INSERT INTO password_reset (user_id, code, created_at) VALUES ($1, $2, $3)
            ON CONFLICT(user_id) DO UPDATE SET code= $2, created_at = $3;",
            user_id,
            code,
            time
        )
        .execute(&self.pool)
        .await?;
        Ok(code)
    }

    pub async fn delete_password_reset_by_user_id(&self, user_id: i32) -> Result<()> {
        query!("DELETE FROM password_reset WHERE user_id = ?", user_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn get_password_reset_by_user_id(
        &self,
        user_id: i32,
    ) -> Result<Option<PasswordResetDAO>> {
        let password_reset = sqlx::query_as(
            "SELECT user_id, code, created_at FROM password_reset WHERE user_id = ?;",
        )
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(password_reset)
    }

    pub async fn delete_expired_password_resets(&self) -> Result<()> {
        let time = Utc::now() - Duration::hours(1);
        query!("DELETE FROM password_reset WHERE created_at < ?", time)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}
