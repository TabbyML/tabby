use anyhow::bail;
use chrono::{DateTime, Utc};
use sqlx::{query, query_as, FromRow};

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
    pub async fn list_user_groups(
        &self,
        user_id: Option<i64>,
    ) -> anyhow::Result<Vec<UserGroupDAO>> {
        let user_groups = if let Some(user_id) = user_id {
            query_as!(
            UserGroupDAO,
            r#"SELECT
                user_groups.id as "id",
                name,
                user_groups.created_at as "created_at: DateTime<Utc>",
                user_groups.updated_at as "updated_at: DateTime<Utc>"
            FROM user_groups LEFT JOIN user_group_memberships ON (user_groups.id = user_group_memberships.user_group_id)
            WHERE user_group_memberships.user_id = ?
            "#,
            user_id
        ).fetch_all(&self.pool)
        .await?
        } else {
            query_as!(
                UserGroupDAO,
                r#"SELECT
                id,
                name,
                created_at as "created_at: DateTime<Utc>",
                updated_at as "updated_at: DateTime<Utc>"
            FROM user_groups
            "#,
            )
            .fetch_all(&self.pool)
            .await?
        };

        Ok(user_groups)
    }

    pub async fn list_user_group_memberships(
        &self,
        user_group_id: i64,
        user_id: Option<i64>,
    ) -> anyhow::Result<Vec<UserGroupMembershipDAO>> {
        let memberships: Vec<_> = query_as!(
            UserGroupMembershipDAO,
            r#"SELECT
              id,
              user_id,
              user_group_id,
              is_group_admin,
              created_at as "created_at: DateTime<Utc>",
              updated_at as "updated_at: DateTime<Utc>"
            FROM user_group_memberships
            WHERE user_group_id = ?1 AND (user_id = ?2 OR ?2 IS NULL)"#,
            user_group_id,
            user_id
        )
        .fetch_all(&self.pool)
        .await?;

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
}
