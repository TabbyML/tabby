use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tabby_db::DbConn;
use tabby_schema::context::ContextService;
use tracing::warn;

use super::helper::Job;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DbMaintainanceJob;

impl Job for DbMaintainanceJob {
    const NAME: &'static str = "db_maintainance";
}

impl DbMaintainanceJob {
    pub async fn cron(
        now: DateTime<Utc>,
        context: Arc<dyn ContextService>,
        db: DbConn,
    ) -> tabby_schema::Result<()> {
        db.delete_expired_token().await?;
        db.delete_expired_password_resets().await?;
        db.delete_expired_ephemeral_threads().await?;

        // Read all active sources
        let active_source_ids = context
            .read(None)
            .await?
            .sources
            .into_iter()
            .map(|x| x.source_id())
            .collect::<Vec<_>>();

        db.delete_unused_source_id_read_access_policy(&active_source_ids)
            .await?;

        Self::data_retention(now, &db).await;
        Ok(())
    }

    async fn data_retention(now: DateTime<Utc>, db: &DbConn) {
        if let Err(e) = db.delete_job_run_before_three_months(now).await {
            warn!(
                "Failed to clean up and retain only the last 3 months of jobs: {:?}",
                e
            );
        }

        if let Err(e) = db.delete_user_events_before_three_months(now).await {
            warn!(
                "Failed to clean up and retain only the last 3 months of user events: {:?}",
                e
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, Utc};
    use tabby_db::DbConn;

    use super::*;

    #[tokio::test]
    async fn test_retention_should_delete() {
        let db = DbConn::new_in_memory().await.unwrap();
        let cases = vec![
            (
                "2024-04-30T12:12:12Z".parse::<DateTime<Utc>>().unwrap(),
                "2024-01-30T12:12:11Z".parse::<DateTime<Utc>>().unwrap(),
            ),
            (
                "2024-04-30T12:12:12Z".parse::<DateTime<Utc>>().unwrap(),
                "2024-01-29T12:12:12Z".parse::<DateTime<Utc>>().unwrap(),
            ),
            (
                "2024-05-01T12:12:12Z".parse::<DateTime<Utc>>().unwrap(),
                "2024-01-31T12:12:11Z".parse::<DateTime<Utc>>().unwrap(),
            ),
        ];

        let user_id = db
            .create_user("user@test.com".to_string(), None, true, None)
            .await
            .unwrap();
        for (now, created) in cases {
            db.create_user_event(
                user_id,
                "test".to_string(),
                created.timestamp_millis() as u128,
                "".to_string(),
            )
            .await
            .unwrap();

            let events = db
                .list_user_events(
                    None,
                    None,
                    false,
                    vec![user_id],
                    created.checked_sub_days(chrono::Days::new(1)).unwrap(),
                    now,
                )
                .await
                .unwrap();
            assert_eq!(events.len(), 1);

            DbMaintainanceJob::data_retention(now, &db).await;

            let events = db
                .list_user_events(
                    None,
                    None,
                    false,
                    vec![user_id],
                    created.checked_sub_days(chrono::Days::new(1)).unwrap(),
                    now,
                )
                .await
                .unwrap();
            assert_eq!(events.len(), 0);
        }
    }

    #[tokio::test]
    async fn test_retention_should_not_delete() {
        let db = DbConn::new_in_memory().await.unwrap();
        let cases = vec![
            (
                "2024-04-30T12:12:12Z".parse::<DateTime<Utc>>().unwrap(),
                "2024-01-31T12:12:12Z".parse::<DateTime<Utc>>().unwrap(),
            ),
            (
                "2024-04-30T12:12:12Z".parse::<DateTime<Utc>>().unwrap(),
                "2024-01-30T12:12:12Z".parse::<DateTime<Utc>>().unwrap(),
            ),
            (
                "2024-04-30T12:12:12Z".parse::<DateTime<Utc>>().unwrap(),
                "2024-04-30T12:12:11Z".parse::<DateTime<Utc>>().unwrap(),
            ),
        ];

        let user_id = db
            .create_user("user@test.com".to_string(), None, true, None)
            .await
            .unwrap();
        for (now, created) in cases {
            db.create_user_event(
                user_id,
                "test".to_string(),
                created.timestamp_millis() as u128,
                "".to_string(),
            )
            .await
            .unwrap();

            let events = db
                .list_user_events(
                    None,
                    None,
                    false,
                    vec![user_id],
                    created.checked_sub_days(chrono::Days::new(1)).unwrap(),
                    now,
                )
                .await
                .unwrap();
            assert_eq!(events.len(), 1);

            DbMaintainanceJob::data_retention(now, &db).await;

            let events = db
                .list_user_events(
                    None,
                    None,
                    false,
                    vec![user_id],
                    created.checked_sub_days(chrono::Days::new(1)).unwrap(),
                    now,
                )
                .await
                .unwrap();
            assert_eq!(events.len(), 1);

            // clean up for next iteration
            db.delete_user_events_before_three_months(
                now.checked_add_months(chrono::Months::new(3)).unwrap(),
            )
            .await
            .unwrap();
        }
    }
}
