use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::{prelude::FromRow, query};
use uuid::Uuid;

use crate::DbConn;

#[derive(FromRow)]
pub struct PasswordResetDAO {
    pub email: String,
    pub code: String,
    pub created_at: DateTime<Utc>,
}

impl DbConn {
    pub async fn create_password_reset(&self, email: String) -> Result<String> {
        let code = Uuid::new_v4().to_string();
        let time = Utc::now();
        query!(
            "INSERT INTO password_reset (email, code, created_at) VALUES ($1, $2, $3)
            ON CONFLICT(email) DO UPDATE SET email = $1, code= $2, created_at = $3;",
            email,
            code,
            time
        )
        .execute(&self.pool)
        .await?;
        Ok(code)
    }

    pub async fn delete_password_reset(&self, email: String) -> Result<()> {
        query!("DELETE FROM password_reset WHERE email = ?", email)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn get_password_reset(&self, email: String) -> Result<PasswordResetDAO> {
        let password_reset =
            sqlx::query_as("SELECT email, code, created_at FROM password_reset WHERE email = ?;")
                .bind(email)
                .fetch_one(&self.pool)
                .await?;
        Ok(password_reset)
    }
}
