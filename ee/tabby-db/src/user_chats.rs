use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::prelude::FromRow;

use crate::{AsSqliteDateTimeString, DbConn};

#[derive(FromRow)]
pub struct UserChatCompletionDailyStatsDAO {
    pub start: DateTime<Utc>,
    pub user_id: i64,
    pub chats: i32,
}

impl DbConn {
    pub async fn compute_chat_daily_stats(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        users: Vec<i64>,
    ) -> Result<Vec<UserChatCompletionDailyStatsDAO>> {
        let users = users
            .iter()
            .map(|u| u.to_string())
            .collect::<Vec<_>>()
            .join(",");

        // Groups stats by day, round all timestamps to the beginning of the day relative to `start`.
        let res = sqlx::query_as::<_, UserChatCompletionDailyStatsDAO>(&format!(
            r#"
            SELECT
                STRFTIME('%F %T', DATE(created_at)) as start,
                user_id,
                COUNT(1) as chats
            FROM user_events
            WHERE created_at >= ?1
              AND created_at < ?2
              AND kind = 'chat_completion'
              AND ({no_selected_users} OR user_id IN ({users}))
            GROUP BY 1, 2
            ORDER BY 1 ASC
            "#,
            no_selected_users = users.is_empty(),
        ))
        .bind(start.as_sqlite_datetime())
        .bind(end.as_sqlite_datetime())
        .fetch_all(&self.pool)
        .await?;

        Ok(res)
    }
}
