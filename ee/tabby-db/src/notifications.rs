use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use sqlx::{prelude::*, query, query_as};

use crate::DbConn;

#[derive(FromRow)]
pub struct NotificationDAO {
    pub id: i64,

    pub recipient: String,
    pub content: String,
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

    pub async fn mark_notification_readed(&self, id: i64, user_id: i64) -> Result<()> {
        query!(
            "INSERT INTO readed_notifications (notification_id, user_id) VALUES (?, ?)",
            id,
            user_id
        )
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
            "recipient = 'all_user' OR recipient = 'admin'"
        } else {
            "recipient = 'all_user'"
        };
        let date_7days_ago = Utc::now() - Duration::days(7);
        let sql = format!(
            r#"
        SELECT notifications.id, notifications.created_at, notifications.updated_at, recipient, content
        FROM notifications LEFT JOIN readed_notifications ON notifications.id = readed_notifications.notification_id
        WHERE ({recipient_clause}) AND notifications.created_at > '{date_7days_ago}' AND readed_notifications.user_id IS NULL  -- notification is not marked as readed
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
