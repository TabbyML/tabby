use chrono::{DateTime, Utc};
use sqlx::query;

use crate::{DbConn, UserGroupDAO};

impl DbConn {
    pub async fn allow_read_source(&self, user_id: i64, source_id: &str) -> anyhow::Result<bool> {
        let is_public_source = query!(
            "select id from source_id_read_access_policies where source_id = ?",
            source_id
        )
        .fetch_optional(&self.pool)
        .await?
        .is_none();

        if is_public_source {
            return Ok(true);
        }

        let row = query!(
            r#"
SELECT user_id
FROM user_groups
    INNER JOIN user_group_memberships ON user_group_memberships.user_group_id = user_groups.id
    INNER JOIN source_id_read_access_policies ON source_id_read_access_policies.user_group_id = user_groups.id
WHERE user_group_memberships.user_id = ? AND source_id = ?
        "#,
            user_id,
            source_id,
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.is_some())
    }

    pub async fn upsert_source_id_read_access_policy(
        &self,
        source_id: &str,
        user_group_id: i64,
    ) -> anyhow::Result<()> {
        query!(
            r#"
INSERT INTO source_id_read_access_policies (source_id, user_group_id) VALUES (?, ?)
ON CONFLICT (source_id, user_group_id) DO UPDATE
  SET updated_at = DATETIME("now")
        "#,
            source_id,
            user_group_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn delete_source_id_read_access_policy(
        &self,
        source_id: &str,
        user_group_id: i64,
    ) -> anyhow::Result<()> {
        let rows_deleted = query!(
            r#"
DELETE FROM source_id_read_access_policies WHERE source_id = ? AND user_group_id = ?
        "#,
            source_id,
            user_group_id
        )
        .execute(&self.pool)
        .await?
        .rows_affected();
        if rows_deleted == 1 {
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "source_id_read_access_policy doesn't exist",
            ))
        }
    }

    pub async fn delete_unused_source_id_read_access_policy(
        &self,
        active_source_ids: &[String],
    ) -> anyhow::Result<usize> {
        let in_clause = active_source_ids
            .iter()
            .map(|s| format!("'{}'", s))
            .collect::<Vec<_>>()
            .join(",");

        let rows_deleted = sqlx::query(&format!(
            "DELETE FROM source_id_read_access_policies WHERE source_id NOT IN ({in_clause})"
        ))
        .execute(&self.pool)
        .await?
        .rows_affected();

        Ok(rows_deleted as usize)
    }

    pub async fn list_source_id_read_access_user_groups(
        &self,
        source_id: &str,
    ) -> anyhow::Result<Vec<UserGroupDAO>> {
        let user_groups = sqlx::query_as!(
            UserGroupDAO,
            r#"SELECT 
                 user_groups.id as "id",
                 name,
                 user_groups.created_at as "created_at: DateTime<Utc>",
                 user_groups.updated_at as "updated_at: DateTime<Utc>"
               FROM source_id_read_access_policies INNER JOIN user_groups ON (source_id_read_access_policies.user_group_id = user_groups.id)
               WHERE source_id = ?
            "#,
            source_id
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(user_groups)
    }
}

#[cfg(test)]
mod tests {
    use crate::DbConn;

    #[tokio::test]
    async fn test_delete_unused_source_id_read_access_policy() {
        let db = DbConn::new_in_memory().await.unwrap();
        let rows_deleted = db
            .delete_unused_source_id_read_access_policy(&["test1".into()])
            .await
            .unwrap();
        assert_eq!(rows_deleted, 0);
    }
}
