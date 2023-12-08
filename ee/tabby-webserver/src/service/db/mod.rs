mod invitations;
mod refresh_tokens;
mod users;

use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use lazy_static::lazy_static;
use rusqlite::params;
use rusqlite_migration::{AsyncMigrations, M};
use tabby_common::path::tabby_root;
use tokio_rusqlite::Connection;

use crate::service::cron::run_offline_job;

lazy_static! {
    static ref MIGRATIONS: AsyncMigrations = AsyncMigrations::new(vec![
        M::up(
            r#"
            CREATE TABLE registration_token (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT (DATETIME('now')),
                updated_at TIMESTAMP DEFAULT (DATETIME('now')),
                CONSTRAINT `idx_token` UNIQUE (`token`)
            );
        "#
        )
        .down("DROP TABLE registration_token"),
        M::up(
            r#"
            CREATE TABLE users (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                email              VARCHAR(150) NOT NULL COLLATE NOCASE,
                password_encrypted VARCHAR(128) NOT NULL,
                is_admin           BOOLEAN NOT NULL DEFAULT 0,
                created_at         TIMESTAMP DEFAULT (DATETIME('now')),
                updated_at         TIMESTAMP DEFAULT (DATETIME('now')),
                CONSTRAINT `idx_email` UNIQUE (`email`)
            );
        "#
        )
        .down("DROP TABLE users"),
        M::up(
            r#"
            CREATE TABLE invitations (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                email              VARCHAR(150) NOT NULL COLLATE NOCASE,
                code               VARCHAR(36) NOT NULL,
                created_at         TIMESTAMP DEFAULT (DATETIME('now')),
                CONSTRAINT `idx_email` UNIQUE (`email`)
                CONSTRAINT `idx_code`  UNIQUE (`code`)
            );
        "#
        )
        .down("DROP TABLE invitations"),
        M::up(
            r#"
            CREATE TABLE refresh_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token VARCHAR(255) NOT NULL COLLATE NOCASE,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT (DATETIME('now')),
                CONSTRAINT `idx_token` UNIQUE (`token`)
            );
        "#
        )
        .down("DROP TABLE refresh_tokens"),
    ]);
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
    #[cfg(test)]
    pub async fn new_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().await?;
        DbConn::init_db(conn).await
    }

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

        let res = Self {
            conn: Arc::new(conn),
        };
        run_offline_job(res.clone());

        Ok(res)
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

#[cfg(test)]
mod tests {

    use super::*;
    use crate::schema::auth::AuthenticationService;

    async fn create_user(conn: &DbConn) -> i32 {
        let email: &str = "test@example.com";
        let password: &str = "123456789";
        conn.create_user(email.to_string(), password.to_string(), true)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn migrations_test() {
        assert!(MIGRATIONS.validate().await.is_ok());
    }

    #[tokio::test]
    async fn test_token() {
        let conn = DbConn::new_in_memory().await.unwrap();
        let token = conn.read_registration_token().await.unwrap();
        assert_eq!(token.len(), 36);
    }

    #[tokio::test]
    async fn test_update_token() {
        let conn = DbConn::new_in_memory().await.unwrap();

        let old_token = conn.read_registration_token().await.unwrap();
        conn.reset_registration_token().await.unwrap();
        let new_token = conn.read_registration_token().await.unwrap();
        assert_eq!(new_token.len(), 36);
        assert_ne!(old_token, new_token);
    }

    #[tokio::test]
    async fn test_is_admin_initialized() {
        let conn = DbConn::new_in_memory().await.unwrap();

        assert!(!conn.is_admin_initialized().await.unwrap());
        create_user(&conn).await;
        assert!(conn.is_admin_initialized().await.unwrap());
    }
}

#[cfg(test)]
mod testutils {
    use super::*;

    pub(crate) async fn create_user(conn: &DbConn) -> i32 {
        let email: &str = "test@example.com";
        let password: &str = "123456789";
        conn.create_user(email.to_string(), password.to_string(), true)
            .await
            .unwrap()
    }
}
