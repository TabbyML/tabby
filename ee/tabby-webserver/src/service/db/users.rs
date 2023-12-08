// db read/write operations for `users` table

use anyhow::Result;
use chrono::{DateTime, Utc};
use rusqlite::{params, OptionalExtension, Row};

use super::DbConn;

#[allow(unused)]
pub struct User {
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,

    pub id: i32,
    pub email: String,
    pub password_encrypted: String,
    pub is_admin: bool,
}

impl User {
    fn select(clause: &str) -> String {
        r#"SELECT id, email, password_encrypted, is_admin, created_at, updated_at FROM users WHERE "#
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
        })
    }
}

impl DbConn {
    pub async fn create_user(
        &self,
        email: String,
        password_encrypted: String,
        is_admin: bool,
    ) -> Result<i32> {
        let res = self
            .conn
            .call(move |c| {
                let mut stmt = c.prepare(
                    r#"INSERT INTO users (email, password_encrypted, is_admin) VALUES (?, ?, ?)"#,
                )?;
                let id = stmt.insert((email, password_encrypted, is_admin))?;
                Ok(id)
            })
            .await?;

        Ok(res as i32)
    }

    pub async fn get_user(&self, id: i32) -> Result<Option<User>> {
        let user = self
            .conn
            .call(move |c| {
                c.query_row(User::select("id = ?").as_str(), params![id], User::from_row)
                    .optional()
            })
            .await?;

        Ok(user)
    }

    pub async fn get_user_by_email(&self, email: &str) -> Result<Option<User>> {
        let email = email.to_owned();
        let user = self
            .conn
            .call(move |c| {
                c.query_row(
                    User::select("email = ?").as_str(),
                    params![email],
                    User::from_row,
                )
                .optional()
            })
            .await?;

        Ok(user)
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
}
