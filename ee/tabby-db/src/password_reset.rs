use anyhow::{anyhow, Result};
use chrono::{Duration, Utc};
use sqlx::{query, query_as};
use uuid::Uuid;

use crate::{DateTimeUtc, DbConn};

pub struct PasswordResetDAO {
    pub user_id: i64,
    pub code: String,
    pub created_at: DateTimeUtc,
}

impl DbConn {
    pub async fn create_password_reset(&self, user_id: i64) -> Result<String> {
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

    pub async fn delete_password_reset_by_user_id(&self, user_id: i64) -> Result<()> {
        query!("DELETE FROM password_reset WHERE user_id = ?", user_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn get_password_reset_by_code(&self, code: &str) -> Result<Option<PasswordResetDAO>> {
        let password_reset = query_as!(
            PasswordResetDAO,
            "SELECT user_id, code, created_at FROM password_reset WHERE code = ?;",
            code
        )
        .fetch_optional(&self.pool)
        .await?;
        Ok(password_reset)
    }

    pub async fn get_password_reset_by_user_id(
        &self,
        user_id: i64,
    ) -> Result<Option<PasswordResetDAO>> {
        let password_reset = query_as!(
            PasswordResetDAO,
            "SELECT user_id, code, created_at FROM password_reset WHERE user_id = ?;",
            user_id
        )
        .fetch_optional(&self.pool)
        .await?;
        Ok(password_reset)
    }

    pub async fn verify_password_reset(&self, code: &str) -> Result<i64> {
        let password_reset = self
            .get_password_reset_by_code(code)
            .await?
            .ok_or_else(|| anyhow!("Invalid code"))?;

        let user_res = self
            .get_user(password_reset.user_id as i32)
            .await?
            .filter(|user| user.active)
            .ok_or_else(|| anyhow!("Invalid code"))?;

        if Utc::now().signed_duration_since(*password_reset.created_at) > Duration::minutes(15) {
            Err(anyhow!("Invalid code"))
        } else {
            Ok(user_res.id as i64)
        }
    }

    #[cfg(any(test, feature = "testutils"))]
    pub async fn mark_password_reset_expired(&self, code: &str) -> Result<()> {
        let timestamp = Utc::now() - Duration::hours(10);
        query!(
            "UPDATE password_reset SET created_at = ? WHERE code = ?;",
            timestamp,
            code
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn delete_expired_password_resets(&self) -> Result<()> {
        let time = Utc::now() - Duration::hours(1);
        query!("DELETE FROM password_reset WHERE created_at < ?", time)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}
