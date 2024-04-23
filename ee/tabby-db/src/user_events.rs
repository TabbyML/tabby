use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use sqlx::{prelude::FromRow, query};
use tabby_db_macros::query_paged_as;

use crate::DbConn;

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
            DateTime::from_timestamp(duration.as_secs() as i64, duration.subsec_nanos())
                .context("Invalid created_at timestamp")?;
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
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<UserEventDAO>> {
        let condition = Some(format!("created_at >= {start} AND created_at < {end}"));
        let events = query_paged_as!(
            UserEventDAO,
            "user_events",
            ["id", "user_id", "kind", "created_at" as "created_at: DateTime<Utc>", "payload"],
            limit,
            skip_id,
            backwards,
            condition
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(events)
    }
}
