use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{DateTime, Months, Utc};
use sqlx::{prelude::FromRow, query};
use tabby_db_macros::query_paged_as;

use crate::{AsSqliteDateTimeString, DbConn};

#[derive(FromRow)]
pub struct UserEventDAO {
    pub id: i64,
    pub user_id: i64,
    pub kind: String,
    pub created_at: DateTime<Utc>,
    pub payload: Vec<u8>,
}

impl DbConn {
    pub async fn create_user_event(
        &self,
        user_id: i64,
        kind: String,
        created_at: u128,
        payload: String,
    ) -> Result<()> {
        let duration = Duration::from_millis(created_at as u64);
        let created_at =
            DateTime::<Utc>::from_timestamp(duration.as_secs() as i64, duration.subsec_nanos())
                .context("Invalid created_at timestamp")?
                .as_sqlite_datetime();
        query!(
            r#"INSERT INTO user_events(user_id, kind, created_at, payload) VALUES (?, ?, ?, ?)"#,
            user_id,
            kind,
            created_at,
            payload
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_user_events(
        &self,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
        users: Vec<i64>,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<UserEventDAO>> {
        let users = users
            .iter()
            .map(|u| u.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let no_selected_users = users.is_empty();
        let condition = Some(format!(
            "created_at >= '{start}' AND created_at < '{end}' AND ({no_selected_users} OR user_id IN ({users}))"
        ));
        let events = query_paged_as!(
            UserEventDAO,
            "user_events",
            ["id", "user_id", "kind", "created_at" as "created_at!: DateTime<Utc>", "payload"],
            limit,
            skip_id,
            backwards,
            condition
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(events)
    }

    pub async fn delete_user_events_before_three_months(
        &self,
        now: DateTime<Utc>,
    ) -> Result<usize> {
        if let Some(three_months_ago) = now.checked_sub_months(Months::new(3)) {
            let three_months_ago = three_months_ago.as_sqlite_datetime();
            let num_deleted = query!(
                "delete FROM user_events WHERE created_at < ?",
                three_months_ago,
            )
            .execute(&self.pool)
            .await?
            .rows_affected();

            Ok(num_deleted as usize)
        } else {
            Ok(0)
        }
    }
}
