// db read/write operations for `users` table

use anyhow::Result;
use chrono::{DateTime, Utc};
use rusqlite::{params, OptionalExtension, Row};
use uuid::Uuid;

use super::DbConn;
use crate::schema;

#[allow(unused)]
pub struct User {
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,

    pub id: i32,
    pub email: String,
    pub password_encrypted: String,
    pub is_admin: bool,

    /// To authenticate IDE extensions / plugins to access code completion / chat api endpoints.
    pub auth_token: String,
}

impl User {
    fn select(clause: &str) -> String {
        r#"SELECT id, email, password_encrypted, is_admin, created_at, updated_at, auth_token FROM users WHERE "#
            .to_owned()
            + clause
    }

    fn from_row(row: &Row<'_>) -> std::result::Result<User, rusqlite::Error> {
        Ok(User {
            id: row.get(0)?,
            email: row.get(1)?,
            password_encrypted: row.get(2)?,
            is_admin: row.get(3)?,
            created_at: row.get(4)?,
            updated_at: row.get(5)?,
            auth_token: row.get(6)?,
        })
    }
}

impl From<User> for schema::User {
    fn from(val: User) -> Self {
        schema::User {
            email: val.email,
            is_admin: val.is_admin,
            auth_token: val.auth_token,
            created_at: val.created_at,
        }
    }
}

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

    pub async fn get_user(&self, id: i32) -> Result<Option<User>> {
        let user = self
            .conn
            .call(move |c| {
                Ok(
                    c.query_row(User::select("id = ?").as_str(), params![id], User::from_row)
                        .optional(),
                )
            })
            .await?;

        Ok(user?)
    }

    pub async fn get_user_by_email(&self, email: &str) -> Result<Option<User>> {
        let email = email.to_owned();
        let user = self
            .conn
            .call(move |c| {
                Ok(c.query_row(
                    User::select("email = ?").as_str(),
                    params![email],
                    User::from_row,
                )
                .optional())
            })
            .await?;

        Ok(user?)
    }

    pub async fn list_admin_users(&self) -> Result<Vec<User>> {
        let users = self
            .conn
            .call(move |c| {
                let mut stmt = c.prepare(&User::select("is_admin"))?;
                let user_iter = stmt.query_map([], User::from_row)?;
                Ok(user_iter.filter_map(|x| x.ok()).collect::<Vec<_>>())
            })
            .await?;

        Ok(users)
    }

    pub async fn list_users(&self) -> Result<Vec<User>> {
        let users = self
            .conn
            .call(move |c| {
                let mut stmt = c.prepare(&User::select("true"))?;
                let user_iter = stmt.query_map([], User::from_row)?;
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
}

fn generate_auth_token() -> String {
    let uuid = Uuid::new_v4().to_string().replace('-', "");
    format!("auth_{}", uuid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::db::testutils::create_user;

    #[tokio::test]
    async fn test_create_user() {
        let conn = DbConn::new_in_memory().await.unwrap();

        let id = create_user(&conn).await;
        let user = conn.get_user(id).await.unwrap().unwrap();
        assert_eq!(user.id, 1);
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
}
