use anyhow::{anyhow, Result};
use sqlx::{prelude::FromRow, query};
use uuid::Uuid;

use super::DbConn;
use crate::{DateTimeUtc, SQLXResultExt};

#[derive(FromRow)]
pub struct InvitationDAO {
    pub id: i64,
    pub email: String,
    pub code: String,

    pub created_at: DateTimeUtc,
}

/// db read/write operations for `invitations` table
impl DbConn {
    pub async fn list_invitations_with_filter(
        &self,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<InvitationDAO>> {
        let query = Self::make_pagination_query(
            "invitations",
            &["id", "email", "code", "created_at"],
            limit,
            skip_id,
            backwards,
        );

        let invitations = sqlx::query_as(&query).fetch_all(&self.pool).await?;

        Ok(invitations)
    }

    pub async fn get_invitation_by_code(&self, code: &str) -> Result<Option<InvitationDAO>> {
        let token = sqlx::query_as!(
            InvitationDAO,
            r#"SELECT id as "id!", email, code, created_at as "created_at!" FROM invitations WHERE code = ?"#,
            code
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(token)
    }

    pub async fn get_invitation_by_email(&self, email: &str) -> Result<Option<InvitationDAO>> {
        let token = sqlx::query_as!(
            InvitationDAO,
            r#"SELECT id as "id!", email, code, created_at as "created_at!" FROM invitations WHERE email = ?"#,
            email
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(token)
    }

    pub async fn create_invitation(&self, email: String) -> Result<InvitationDAO> {
        if self.get_user_by_email(&email).await?.is_some() {
            return Err(anyhow!("User already registered"));
        }

        let code = Uuid::new_v4().to_string();
        let created_at = chrono::offset::Utc::now().into();
        let res = query!(
            "INSERT INTO invitations (email, code, created_at) VALUES (?, ?, ?)",
            email,
            code,
            created_at
        )
        .execute(&self.pool)
        .await;

        let res = res.unique_error("Failed to create invitation, email already exists")?;
        let id = res.last_insert_rowid();

        Ok(InvitationDAO {
            id,
            email,
            code,
            created_at,
        })
    }

    pub async fn delete_invitation(&self, id: i64) -> Result<i64> {
        let res = query!("DELETE FROM invitations WHERE id = ?", id)
            .execute(&self.pool)
            .await?;
        if res.rows_affected() != 1 {
            return Err(anyhow!("failed to delete invitation"));
        }

        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_invitations() {
        let conn = DbConn::new_in_memory().await.unwrap();

        let email = "hello@example.com".to_owned();
        conn.create_invitation(email).await.unwrap();

        let invitations = conn
            .list_invitations_with_filter(None, None, false)
            .await
            .unwrap();
        assert_eq!(1, invitations.len());

        assert!(Uuid::parse_str(&invitations[0].code).is_ok());
        let invitation = conn
            .get_invitation_by_code(&invitations[0].code)
            .await
            .ok()
            .flatten()
            .unwrap();
        assert_eq!(invitation.id, invitations[0].id);

        conn.delete_invitation(invitations[0].id).await.unwrap();

        let invitations = conn
            .list_invitations_with_filter(None, None, false)
            .await
            .unwrap();
        assert!(invitations.is_empty());
    }
}
