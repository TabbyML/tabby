use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use sqlx::{prelude::*, query, query_as};

use crate::DbConn;

const NOTIFICATION_RECIPIENT_ALL_USER: &str = "all_user";
const NOTIFICATION_RECIPIENT_ADMIN: &str = "admin";

#[derive(FromRow)]
pub struct NotificationDAO {
    pub id: i64,

    pub recipient: String,
    pub content: String,
    pub read: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl DbConn {
    pub async fn create_notification(&self, recipient: &str, content: &str) -> Result<i64> {
        let res = query!(
            "INSERT INTO notifications (recipient, content) VALUES (?, ?)",
            recipient,
            content
        )
        .execute(&self.pool)
        .await?;

        Ok(res.last_insert_rowid())
    }

    pub async fn mark_notification_read(&self, id: i64, user_id: i64) -> Result<()> {
        query!(
            "INSERT INTO read_notifications (notification_id, user_id)
             VALUES (?, ?)
             ON CONFLICT (notification_id, user_id)
             DO NOTHING",
            id,
            user_id,
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn mark_all_notifications_read_by_user(&self, user_id: i64) -> Result<()> {
        let user = self
            .get_user(user_id)
            .await?
            .context("User doesn't exist")?;
        let recipient_clause = if user.is_admin {
            format!(
                "recipient = '{NOTIFICATION_RECIPIENT_ALL_USER}' OR recipient = '{NOTIFICATION_RECIPIENT_ADMIN}'"
            )
        } else {
            format!("recipient = '{NOTIFICATION_RECIPIENT_ALL_USER}'")
        };

        let query = format!(
            r#"
INSERT INTO read_notifications (notification_id, user_id)
SELECT
    notifications.id,
    ?
FROM
    notifications
LEFT JOIN
    read_notifications
ON
    notifications.id = read_notifications.notification_id
    AND read_notifications.user_id = ?
WHERE
    ({recipient_clause})
    AND read_notifications.notification_id IS NULL;
        "#
        );

        sqlx::query(&query)
            .bind(user_id)
            .bind(user_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    pub async fn list_notifications_within_7days(
        &self,
        user_id: i64,
    ) -> Result<Vec<NotificationDAO>> {
        let user = self
            .get_user(user_id)
            .await?
            .context("User doesn't exist")?;
        let recipient_clause = if user.is_admin {
            format!(
                "recipient = '{NOTIFICATION_RECIPIENT_ALL_USER}' OR recipient = '{NOTIFICATION_RECIPIENT_ADMIN}'"
            )
        } else {
            format!("recipient = '{NOTIFICATION_RECIPIENT_ALL_USER}'")
        };
        let date_7days_ago = Utc::now() - Duration::days(7);
        let sql = format!(
            r#"
SELECT
    notifications.id,
    notifications.created_at,
    notifications.updated_at,
    notifications.recipient,
    notifications.content,
    CASE
        WHEN read_notifications.user_id = '{user_id}' THEN 1
        ELSE 0
    END AS read
FROM
    notifications
LEFT JOIN
    read_notifications
ON
    notifications.id = read_notifications.notification_id
    AND read_notifications.user_id = '{user_id}'
WHERE
    ({recipient_clause})
    AND notifications.created_at > '{date_7days_ago}'
        "#
        );
        let notifications = query_as(&sql).fetch_all(&self.pool).await?;
        Ok(notifications)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutils;

    /// Smoke test to ensure sql query is valid, actual functionality test shall happens at service level.
    #[tokio::test]
    async fn smoketest_list_notifications() {
        let db = DbConn::new_in_memory().await.unwrap();
        let user1 = testutils::create_user(&db).await;
        let notifications = db.list_notifications_within_7days(user1).await.unwrap();
        assert!(notifications.is_empty())
    }
}
