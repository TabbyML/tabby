use anyhow::bail;
use chrono::{DateTime, Utc};
use sqlx::{query, query_as, FromRow};
use tabby_db_macros::query_paged_as;

use crate::DbConn;

#[derive(FromRow)]
pub struct UserGroupDAO {
    pub id: i64,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(FromRow)]
pub struct UserGroupMembershipDAO {
    pub id: i64,
    pub user_id: i64,
    pub user_group_id: i64,
    pub is_group_admin: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

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

    pub async fn list_user_groups(
        &self,
        user_id: Option<i64>,
    ) -> anyhow::Result<Vec<UserGroupDAO>> {
        let user_groups = query_as!(
            UserGroupDAO,
            r#"SELECT
                user_groups.id as "id",
                name,
                user_groups.created_at as "created_at: DateTime<Utc>",
                user_groups.updated_at as "updated_at: DateTime<Utc>"
            FROM user_groups LEFT JOIN user_group_memberships ON (user_group_memberships.user_group_id = user_groups.id)
            WHERE (user_group_memberships.user_id = ?1 OR NULL = ?1)
            "#,
            user_id
        ).fetch_all(&self.pool)
        .await?;

        Ok(user_groups)
    }

    pub async fn list_user_group_memberships(
        &self,
        id: i64,
        user_id: Option<i64>,
    ) -> anyhow::Result<Vec<UserGroupMembershipDAO>> {
        let memberships: Vec<_> = if let Some(user_id) = user_id {
            query_as!(
                UserGroupMembershipDAO,
                r#"SELECT
              id,
              user_id,
              user_group_id,
              is_group_admin,
              created_at as "created_at: DateTime<Utc>",
              updated_at as "updated_at: DateTime<Utc>"
            FROM user_group_memberships
            WHERE user_group_id = ? AND user_id = ?"#,
                id,
                user_id
            )
            .fetch_all(&self.pool).await?
        } else {
            query_as!(
                UserGroupMembershipDAO,
                r#"SELECT
              id,
              user_id,
              user_group_id,
              is_group_admin,
              created_at as "created_at: DateTime<Utc>",
              updated_at as "updated_at: DateTime<Utc>"
            FROM user_group_memberships
            WHERE user_group_id = ?"#,
                id,
            )
            .fetch_all(&self.pool).await?
        };

        Ok(memberships)
    }

    pub async fn create_user_group(&self, name: &str) -> anyhow::Result<i64> {
        let id = query!("INSERT INTO user_groups (name) VALUES (?)", name)
            .execute(&self.pool)
            .await?
            .last_insert_rowid();

        Ok(id)
    }

    pub async fn delete_user_group(&self, id: i64) -> anyhow::Result<()> {
        let res = query!("DELETE FROM user_groups WHERE id = ?", id)
            .execute(&self.pool)
            .await?;

        if res.rows_affected() != 1 {
            bail!("User group not found");
        }

        Ok(())
    }

    pub async fn is_user_group_admin(
        &self,
        user_id: i64,
        user_group_id: i64,
    ) -> anyhow::Result<bool> {
        struct Result {
            is_group_admin: bool,
        }

        let res = query_as!(
            Result,
            r#"SELECT
                is_group_admin
            FROM user_group_memberships WHERE user_id = ? AND user_group_id = ? AND is_group_admin"#,
            user_id,
            user_group_id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(res.is_some_and(|x| x.is_group_admin))
    }

    pub async fn upsert_user_group_membership(
        &self,
        user_id: i64,
        user_group_id: i64,
        is_group_admin: bool,
    ) -> anyhow::Result<()> {
        let _ = query!(
            r#"
INSERT INTO user_group_memberships (user_id, user_group_id, is_group_admin) VALUES (?, ?, ?)
ON CONFLICT (user_id, user_group_id) DO UPDATE
  SET is_group_admin = excluded.is_group_admin, updated_at = DATETIME('now')
        "#,
            user_id,
            user_group_id,
            is_group_admin
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn delete_user_group_membership(
        &self,
        user_id: i64,
        user_group_id: i64,
    ) -> anyhow::Result<()> {
        let rows_deleted = query!(
            r#"
DELETE FROM user_group_memberships WHERE user_id = ? AND user_group_id = ?
        "#,
            user_id,
            user_group_id
        )
        .execute(&self.pool)
        .await?
        .rows_affected();

        if rows_deleted == 1 {
            Ok(())
        } else {
            Err(anyhow::anyhow!("User group membership not found"))
        }
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
}
