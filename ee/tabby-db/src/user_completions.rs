use std::time::Duration;

use anyhow::{Context, Result};
use chrono::DateTime;
use sqlx::{prelude::FromRow, query};

use crate::{DateTimeUtc, DbConn};

#[derive(FromRow)]
pub struct UserCompletionDAO {
    pub user_id: i64,
    pub completion_id: String,
    pub language: String,

    pub views: i64,
    pub selects: i64,
    pub dismisses: i64,

    pub created_at: DateTimeUtc,
    pub updated_at: DateTimeUtc,
}

impl DbConn {
    pub async fn create_user_completion(
        &self,
        ts: u128,
        user_id: i64,
        completion_id: String,
        language: String,
    ) -> Result<i32> {
        let duration = Duration::from_millis(ts as u64);
        let created_at =
            DateTime::from_timestamp(duration.as_secs() as i64, duration.subsec_nanos())
                .context("Invalid created_at timestamp")?;
        let res = query!(
            "INSERT INTO user_completions (user_id, completion_id, language, created_at) VALUES (?, ?, ?, ?);",
            user_id,
            completion_id,
            language,
            created_at
        )
        .execute(&self.pool)
        .await?;
        Ok(res.last_insert_rowid() as i32)
    }

    pub async fn add_to_user_completion(
        &self,
        ts: u128,
        completion_id: &str,
        views: i64,
        selects: i64,
        dismisses: i64,
    ) -> Result<()> {
        let duration = Duration::from_millis(ts as u64);
        let updated_at =
            DateTime::from_timestamp(duration.as_secs() as i64, duration.subsec_nanos())
                .context("Invalid updated_at timestamp")?;
        query!("UPDATE user_completions SET views = views + ?, selects = selects + ?, dismisses = dismisses + ?, updated_at = ? WHERE completion_id = ?",
            views, selects, dismisses, updated_at, completion_id).execute(&self.pool).await?;
        Ok(())
    }

    #[cfg(any(test, feature = "testutils"))]
    pub async fn fetch_one_user_completion(&self) -> Result<Option<UserCompletionDAO>> {
        Ok(
            sqlx::query_as!(UserCompletionDAO, "SELECT user_id, completion_id, language, created_at, updated_at, views, selects, dismisses FROM user_completions")
                .fetch_optional(&self.pool)
                .await?,
        )
    }
}
