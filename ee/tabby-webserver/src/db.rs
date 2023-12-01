use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use lazy_static::lazy_static;
use rusqlite::{params, OptionalExtension, Row};
use rusqlite_migration::{AsyncMigrations, M};
use tabby_common::path::tabby_root;
use tokio_rusqlite::Connection;

lazy_static! {
    static ref MIGRATIONS: AsyncMigrations = AsyncMigrations::new(vec![
        M::up(
            r#"
            CREATE TABLE IF NOT EXISTS registration_token (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT (DATETIME('now')),
                updated_at TIMESTAMP DEFAULT (DATETIME('now')),
                CONSTRAINT `idx_token` UNIQUE (`token`)
            );
        "#
        ),
        M::up(
            r#"
            CREATE TABLE IF NOT EXISTS users (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                email              VARCHAR(150) NOT NULL COLLATE NOCASE,
                password_encrypted VARCHAR(128) NOT NULL,
                is_admin           BOOLEAN NOT NULL DEFAULT 0,
                created_at         TIMESTAMP DEFAULT (DATETIME('now')),
                updated_at         TIMESTAMP DEFAULT (DATETIME('now')),
                CONSTRAINT `idx_email` UNIQUE (`email`)
            );
        "#
        ),
    ]);
}

#[allow(unused)]
pub struct User {
    created_at: String,
    updated_at: String,

    pub id: u32,
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

async fn db_path() -> Result<PathBuf> {
    let db_dir = tabby_root().join("ee");
    tokio::fs::create_dir_all(db_dir.clone()).await?;
    Ok(db_dir.join("db.sqlite"))
}

#[derive(Clone)]
pub struct DbConn {
    conn: Arc<Connection>,
}

impl DbConn {
    pub async fn new() -> Result<Self> {
        let db_path = db_path().await?;
        let conn = Connection::open(db_path).await?;
        Self::init_db(conn).await
    }

    /// Initialize database, create tables and insert first token if not exist
    async fn init_db(mut conn: Connection) -> Result<Self> {
        MIGRATIONS.to_latest(&mut conn).await?;

        let token = uuid::Uuid::new_v4().to_string();
        conn.call(move |c| {
            c.execute(
                r#"INSERT OR IGNORE INTO registration_token (id, token) VALUES (1, ?)"#,
                params![token],
            )
        })
        .await?;

        Ok(Self {
            conn: Arc::new(conn),
        })
    }
}

/// db read/write operations for `registration_token` table
impl DbConn {
    /// Query token from database.
    /// Since token is global unique for each tabby server, by right there's only one row in the table.
    pub async fn read_registration_token(&self) -> Result<String> {
        let token = self
            .conn
            .call(|conn| {
                conn.query_row(
                    r#"SELECT token FROM registration_token WHERE id = 1"#,
                    [],
                    |row| row.get(0),
                )
            })
            .await?;

        Ok(token)
    }

    /// Update token in database.
    pub async fn reset_registration_token(&self) -> Result<String> {
        let token = uuid::Uuid::new_v4().to_string();
        let result = token.clone();
        let updated_at = chrono::Utc::now().timestamp() as u32;

        let res = self
            .conn
            .call(move |conn| {
                conn.execute(
                    r#"UPDATE registration_token SET token = ?, updated_at = ? WHERE id = 1"#,
                    params![token, updated_at],
                )
            })
            .await?;
        if res != 1 {
            return Err(anyhow::anyhow!("failed to update token"));
        }

        Ok(result)
    }
}

/// db read/write operations for `users` table
impl DbConn {
    pub async fn create_user(
        &self,
        email: String,
        password_encrypted: String,
        is_admin: bool,
    ) -> Result<()> {
        let res = self
            .conn
            .call(move |c| {
                c.execute(
                    r#"INSERT INTO users (email, password_encrypted, is_admin) VALUES (?, ?, ?)"#,
                    params![email, password_encrypted, is_admin],
                )
            })
            .await?;
        if res != 1 {
            return Err(anyhow::anyhow!("failed to create user"));
        }

        Ok(())
    }

    pub async fn get_user_by_email(&self, email: &str) -> Result<Option<User>> {
        let email = email.to_string();
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

    pub async fn get_admin_users(&self) -> Result<Vec<User>> {
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
    use crate::server::auth::AuthenticationService;

    use super::*;

    async fn new_in_memory() -> Result<DbConn> {
        let conn = Connection::open_in_memory().await?;
        DbConn::init_db(conn).await
    }

    async fn create_admin_user(conn: &DbConn) -> String {
        let email = "test@example.com";
        let passwd = "123456";
        let is_admin = true;
        conn.create_user(email.to_string(), passwd.to_string(), is_admin)
            .await
            .unwrap();
        email.to_owned()
    }

    #[tokio::test]
    async fn migrations_test() {
        assert!(MIGRATIONS.validate().await.is_ok());
    }

    #[tokio::test]
    async fn test_token() {
        let conn = new_in_memory().await.unwrap();
        let token = conn.read_registration_token().await.unwrap();
        assert_eq!(token.len(), 36);
    }

    #[tokio::test]
    async fn test_update_token() {
        let conn = new_in_memory().await.unwrap();

        let old_token = conn.read_registration_token().await.unwrap();
        conn.reset_registration_token().await.unwrap();
        let new_token = conn.read_registration_token().await.unwrap();
        assert_eq!(new_token.len(), 36);
        assert_ne!(old_token, new_token);
    }

    #[tokio::test]
    async fn test_create_user() {
        let conn = new_in_memory().await.unwrap();

        let email = create_admin_user(&conn).await;
        let user = conn.get_user_by_email(&email).await.unwrap().unwrap();
        assert_eq!(user.id, 1);
    }

    #[tokio::test]
    async fn test_get_user_by_email() {
        let conn = new_in_memory().await.unwrap();

        let email = "hello@example.com";
        let user = conn.get_user_by_email(email).await.unwrap();

        assert!(user.is_none());
    }

    #[tokio::test]
    async fn test_is_admin_initialized() {
        let conn = new_in_memory().await.unwrap();

        assert!(!conn.is_admin_initialized().await.unwrap());
        create_admin_user(&conn).await;
        assert!(conn.is_admin_initialized().await.unwrap());
    }
}
