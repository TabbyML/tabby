use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::{prelude::FromRow, query};

use crate::DbConn;

#[derive(FromRow)]
pub struct UserCompletionDAO {
    pub user_id: i32,
    pub completion_id: String,
    pub language: String,

    pub views: i64,
    pub selects: i64,
    pub dismisses: i64,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl DbConn {
    pub async fn create_user_completion(
        &self,
        user_id: i32,
        completion_id: String,
        language: String,
    ) -> Result<i32> {
        let res = query!(
            "INSERT INTO user_completions (user_id, completion_id, language) VALUES (?, ?, ?);",
            user_id,
            completion_id,
            language
        )
        .execute(&self.pool)
        .await?;
        Ok(res.last_insert_rowid() as i32)
    }

    pub async fn add_to_user_completion(
        &self,
        completion_id: &str,
        views: i64,
        selects: i64,
        dismisses: i64,
    ) -> Result<()> {
        query!("UPDATE user_completions SET views = views + ?, selects = selects + ?, dismisses = dismisses + ? WHERE completion_id = ?",
            views, selects, dismisses, completion_id).execute(&self.pool).await?;
        Ok(())
    }

    #[cfg(any(test, feature = "testutils"))]
    pub async fn fetch_one_user_completion(&self) -> Result<Option<UserCompletionDAO>> {
        Ok(sqlx::query_as("SELECT * FROM user_completions")
            .fetch_optional(&self.pool)
            .await?)
    }
}
