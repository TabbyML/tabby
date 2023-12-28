use anyhow::{anyhow, Result};
use rusqlite::{params, OptionalExtension, Row};
use uuid::Uuid;

use super::DbConn;
use crate::schema::auth;

pub struct InvitationDAO {
    pub id: i32,
    pub email: String,
    pub code: String,

    pub created_at: String,
}

impl InvitationDAO {
    fn from_row(row: &Row<'_>) -> std::result::Result<Self, rusqlite::Error> {
        Ok(Self {
            id: row.get(0)?,
            email: row.get(1)?,
            code: row.get(2)?,
            created_at: row.get(3)?,
        })
    }
}

impl From<InvitationDAO> for auth::InvitationNext {
    fn from(val: InvitationDAO) -> Self {
        Self {
            id: juniper::ID::new(val.id.to_string()),
            email: val.email,
            code: val.code,
            created_at: val.created_at,
        }
    }
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

        let invitations = self
            .conn
            .call(move |c| {
                let mut stmt = c.prepare(&query)?;
                let invit_iter = stmt.query_map([], InvitationDAO::from_row)?;
                Ok(invit_iter.filter_map(|x| x.ok()).collect::<Vec<_>>())
            })
            .await?;

        Ok(invitations)
    }

    pub async fn get_invitation_by_code(&self, code: &str) -> Result<Option<InvitationDAO>> {
        let code = code.to_owned();
        let token = self
            .conn
            .call(|conn| {
                Ok(conn
                    .query_row(
                        r#"SELECT id, email, code, created_at FROM invitations WHERE code = ?"#,
                        [code],
                        InvitationDAO::from_row,
                    )
                    .optional())
            })
            .await?;

        Ok(token?)
    }

    pub async fn create_invitation(&self, email: String) -> Result<i32> {
        if self.get_user_by_email(&email).await?.is_some() {
            return Err(anyhow!("User already registered"));
        }

        let code = Uuid::new_v4().to_string();
        let res = self
            .conn
            .call(move |c| {
                let mut stmt =
                    c.prepare(r#"INSERT INTO invitations (email, code) VALUES (?, ?)"#)?;
                let rowid = stmt.insert((email, code))?;
                Ok(rowid)
            })
            .await;

        match res {
            Err(tokio_rusqlite::Error::Rusqlite(rusqlite::Error::SqliteFailure(err, msg))) => {
                if err.code == rusqlite::ErrorCode::ConstraintViolation {
                    Err(anyhow!("Failed to create invitation, email already exists"))
                } else {
                    Err(rusqlite::Error::SqliteFailure(err, msg).into())
                }
            }
            Err(err) => Err(err.into()),
            Ok(rowid) => Ok(rowid as i32),
        }
    }

    pub async fn delete_invitation(&self, id: i32) -> Result<i32> {
        let res = self
            .conn
            .call(move |c| Ok(c.execute(r#"DELETE FROM invitations WHERE id = ?"#, params![id])))
            .await?;
        if res != Ok(1) {
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
