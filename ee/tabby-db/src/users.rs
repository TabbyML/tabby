use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, OptionalExtension, Row};
use uuid::Uuid;

use super::DbConn;

#[allow(unused)]
pub struct UserDAO {
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,

    pub id: i32,
    pub email: String,
    pub password_encrypted: String,
    pub is_admin: bool,

    /// To authenticate IDE extensions / plugins to access code completion / chat api endpoints.
    pub auth_token: String,
    pub active: bool,
}

impl UserDAO {
    fn select(clause: &str) -> String {
        r#"SELECT id, email, password_encrypted, is_admin, created_at, updated_at, auth_token, active FROM users WHERE "#
            .to_owned()
            + clause
    }

    fn from_row(row: &Row<'_>) -> std::result::Result<UserDAO, rusqlite::Error> {
        Ok(UserDAO {
            id: row.get(0)?,
            email: row.get(1)?,
            password_encrypted: row.get(2)?,
            is_admin: row.get(3)?,
            created_at: row.get(4)?,
            updated_at: row.get(5)?,
            auth_token: row.get(6)?,
            active: row.get(7)?,
        })
    }
}

/// db read/write operations for `users` table
impl DbConn {
    pub async fn create_user(
        &self,
        email: String,
        password_encrypted: String,
        is_admin: bool,
    ) -> Result<i32> {
        self.create_user_impl(email, password_encrypted, is_admin, None)
            .await
    }

    pub async fn create_user_with_invitation(
        &self,
        email: String,
        password_encrypted: String,
        is_admin: bool,
        invitation_id: i32,
    ) -> Result<i32> {
        self.create_user_impl(email, password_encrypted, is_admin, Some(invitation_id))
            .await
    }

    async fn create_user_impl(
        &self,
        email: String,
        password_encrypted: String,
        is_admin: bool,
        invitation_id: Option<i32>,
    ) -> Result<i32> {
        let res = self
            .conn
            .call(move |c| {
                let tx = c.transaction()?;

                if let Some(invitation_id) = invitation_id {
                    tx.execute("DELETE FROM invitations WHERE id = ?", params![invitation_id])?;
                }

                let id = {
                    let mut stmt = tx.prepare(
                        r#"INSERT INTO users (email, password_encrypted, is_admin, auth_token) VALUES (?, ?, ?, ?)"#,
                    )?;
                    stmt.insert((email, password_encrypted, is_admin, generate_auth_token()))?
                };

                tx.commit()?;
                Ok(id)
            })
            .await?;

        Ok(res as i32)
    }

    pub async fn get_user(&self, id: i32) -> Result<Option<UserDAO>> {
        let user = self
            .conn
            .call(move |c| {
                Ok(c.query_row(
                    UserDAO::select("id = ?").as_str(),
                    params![id],
                    UserDAO::from_row,
                )
                .optional())
            })
            .await?;

        Ok(user?)
    }

    pub async fn get_user_by_email(&self, email: &str) -> Result<Option<UserDAO>> {
        let email = email.to_owned();
        let user = self
            .conn
            .call(move |c| {
                Ok(c.query_row(
                    UserDAO::select("email = ?").as_str(),
                    params![email],
                    UserDAO::from_row,
                )
                .optional())
            })
            .await?;

        Ok(user?)
    }

    pub async fn list_admin_users(&self) -> Result<Vec<UserDAO>> {
        let users = self
            .conn
            .call(move |c| {
                let mut stmt = c.prepare(&UserDAO::select("is_admin"))?;
                let user_iter = stmt.query_map([], UserDAO::from_row)?;
                Ok(user_iter.filter_map(|x| x.ok()).collect::<Vec<_>>())
            })
            .await?;

        Ok(users)
    }

    pub async fn list_users_with_filter(
        &self,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<UserDAO>> {
        let query = Self::make_pagination_query(
            "users",
            &[
                "id",
                "email",
                "password_encrypted",
                "is_admin",
                "created_at",
                "updated_at",
                "auth_token",
                "active",
            ],
            limit,
            skip_id,
            backwards,
        );

        let users = self
            .conn
            .call(move |c| {
                let mut stmt = c.prepare(&query)?;
                let user_iter = stmt.query_map([], UserDAO::from_row)?;
                Ok(user_iter.filter_map(|x| x.ok()).collect::<Vec<_>>())
            })
            .await?;

        Ok(users)
    }

    pub async fn verify_auth_token(&self, token: &str) -> bool {
        let token = token.to_owned();
        let id: Result<Result<i32, _>, _> = self
            .conn
            .call(move |c| {
                Ok(c.query_row(
                    r#"SELECT id FROM users WHERE auth_token = ?"#,
                    params![token],
                    |row| row.get(0),
                ))
            })
            .await;
        matches!(id, Ok(Ok(_)))
    }

    pub async fn reset_user_auth_token_by_email(&self, email: &str) -> Result<()> {
        let email = email.to_owned();
        let updated_at = chrono::Utc::now();
        self.conn
            .call(move |c| {
                let mut stmt = c.prepare(
                    r#"UPDATE users SET auth_token = ?, updated_at = ? WHERE email = ?"#,
                )?;
                stmt.execute((generate_auth_token(), updated_at, email))?;
                Ok(())
            })
            .await?;

        Ok(())
    }

    pub async fn update_user_active(&self, id: i32, active: bool) -> Result<()> {
        let new_active = self
            .conn
            .call(move |c| {
                c.execute("UPDATE users SET active=? WHERE id=?", (active, id))?;
                let new_active: bool =
                    c.query_row("SELECT active FROM users WHERE id=?", [id], |r| r.get(0))?;
                Ok(new_active)
            })
            .await?;
        if new_active != active {
            Err(anyhow!("user active status was not changed"))
        } else {
            Ok(())
        }
    }
}

fn generate_auth_token() -> String {
    let uuid = Uuid::new_v4().to_string().replace('-', "");
    format!("auth_{}", uuid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutils::create_user;

    #[tokio::test]
    async fn test_create_user() {
        let conn = DbConn::new_in_memory().await.unwrap();

        let id = create_user(&conn).await;
        let user = conn.get_user(id).await.unwrap().unwrap();
        assert_eq!(user.id, 1);
    }

    #[tokio::test]
    async fn test_set_active() {
        let conn = DbConn::new_in_memory().await.unwrap();
        let id = create_user(&conn).await;

        assert!(conn.get_user(id).await.unwrap().unwrap().active);

        conn.update_user_active(id, false).await.unwrap();

        assert!(!conn.get_user(id).await.unwrap().unwrap().active);
    }

    #[tokio::test]
    async fn test_get_user_by_email() {
        let conn = DbConn::new_in_memory().await.unwrap();

        let email = "hello@example.com";
        let user = conn.get_user_by_email(email).await.unwrap();

        assert!(user.is_none());
    }

    #[tokio::test]
    async fn test_auth_token() {
        let conn = DbConn::new_in_memory().await.unwrap();
        let id = create_user(&conn).await;

        let user = conn.get_user(id).await.unwrap().unwrap();

        assert!(!conn.verify_auth_token("abcd").await);

        assert!(conn.verify_auth_token(&user.auth_token).await);

        conn.reset_user_auth_token_by_email(&user.email)
            .await
            .unwrap();
        let new_user = conn.get_user(id).await.unwrap().unwrap();
        assert_eq!(user.email, new_user.email);
        assert_ne!(user.auth_token, new_user.auth_token);
    }

    #[tokio::test]
    async fn test_list_users_with_filter() {
        let conn = DbConn::new_in_memory().await.unwrap();

        let empty: Vec<i32> = vec![];
        let to_ids = |users: Vec<UserDAO>| users.into_iter().map(|u| u.id).collect::<Vec<_>>();

        // empty
        // forwards
        assert_eq!(
            empty,
            to_ids(
                conn.list_users_with_filter(None, None, false)
                    .await
                    .unwrap()
            )
        );
        assert_eq!(
            empty,
            to_ids(
                conn.list_users_with_filter(Some(2), None, false)
                    .await
                    .unwrap()
            )
        );
        assert_eq!(
            empty,
            to_ids(
                conn.list_users_with_filter(None, Some(1), false)
                    .await
                    .unwrap()
            )
        );
        assert_eq!(
            empty,
            to_ids(
                conn.list_users_with_filter(Some(2), Some(1), false)
                    .await
                    .unwrap()
            )
        );
        // backwards
        assert_eq!(
            empty,
            to_ids(conn.list_users_with_filter(None, None, true).await.unwrap())
        );
        assert_eq!(
            empty,
            to_ids(
                conn.list_users_with_filter(Some(2), None, true)
                    .await
                    .unwrap()
            )
        );
        assert_eq!(
            empty,
            to_ids(
                conn.list_users_with_filter(None, Some(1), true)
                    .await
                    .unwrap()
            )
        );
        assert_eq!(
            empty,
            to_ids(
                conn.list_users_with_filter(Some(1), Some(1), true)
                    .await
                    .unwrap()
            )
        );

        let id1 = conn
            .create_user("use1@example.com".into(), "123456".into(), false)
            .await
            .unwrap();

        // one user
        // forwards
        assert_eq!(
            vec![id1],
            to_ids(
                conn.list_users_with_filter(None, None, false)
                    .await
                    .unwrap()
            )
        );
        assert_eq!(
            vec![id1],
            to_ids(
                conn.list_users_with_filter(Some(2), None, false)
                    .await
                    .unwrap()
            )
        );
        assert_eq!(
            empty,
            to_ids(
                conn.list_users_with_filter(None, Some(1), false)
                    .await
                    .unwrap()
            )
        );
        assert_eq!(
            empty,
            to_ids(
                conn.list_users_with_filter(Some(2), Some(1), false)
                    .await
                    .unwrap()
            )
        );
        // backwards
        assert_eq!(
            vec![id1],
            to_ids(conn.list_users_with_filter(None, None, true).await.unwrap())
        );
        assert_eq!(
            vec![id1],
            to_ids(
                conn.list_users_with_filter(Some(2), None, true)
                    .await
                    .unwrap()
            )
        );
        assert_eq!(
            empty,
            to_ids(
                conn.list_users_with_filter(None, Some(1), true)
                    .await
                    .unwrap()
            )
        );
        assert_eq!(
            empty,
            to_ids(
                conn.list_users_with_filter(Some(1), Some(1), true)
                    .await
                    .unwrap()
            )
        );

        let id2 = conn
            .create_user("use2@example.com".into(), "123456".into(), false)
            .await
            .unwrap();
        let id3 = conn
            .create_user("use3@example.com".into(), "123456".into(), false)
            .await
            .unwrap();
        let id4 = conn
            .create_user("use4@example.com".into(), "123456".into(), false)
            .await
            .unwrap();
        let id5 = conn
            .create_user("use5@example.com".into(), "123456".into(), false)
            .await
            .unwrap();

        // multiple users
        // forwards
        assert_eq!(
            vec![id1, id2, id3, id4, id5],
            to_ids(
                conn.list_users_with_filter(None, None, false)
                    .await
                    .unwrap()
            )
        );
        assert_eq!(
            vec![id1, id2],
            to_ids(
                conn.list_users_with_filter(Some(2), None, false)
                    .await
                    .unwrap()
            )
        );
        assert_eq!(
            vec![id3, id4, id5],
            to_ids(
                conn.list_users_with_filter(None, Some(2), false)
                    .await
                    .unwrap()
            )
        );
        assert_eq!(
            vec![id3, id4],
            to_ids(
                conn.list_users_with_filter(Some(2), Some(2), false)
                    .await
                    .unwrap()
            )
        );
        // backwards
        assert_eq!(
            vec![id1, id2, id3, id4, id5],
            to_ids(conn.list_users_with_filter(None, None, true).await.unwrap())
        );
        assert_eq!(
            vec![id4, id5],
            to_ids(
                conn.list_users_with_filter(Some(2), None, true)
                    .await
                    .unwrap()
            )
        );
        assert_eq!(
            vec![id1, id2, id3],
            to_ids(
                conn.list_users_with_filter(None, Some(4), true)
                    .await
                    .unwrap()
            )
        );
        assert_eq!(
            vec![id2, id3],
            to_ids(
                conn.list_users_with_filter(Some(2), Some(4), true)
                    .await
                    .unwrap()
            )
        );
    }
}
