mod invitations;
mod job_runs;
mod refresh_tokens;
mod users;

use anyhow::Result;
use include_dir::{include_dir, Dir};
pub use job_runs::JobRun;
use lazy_static::lazy_static;
use rusqlite::params;
use rusqlite_migration::AsyncMigrations;
use tokio_rusqlite::Connection;

use crate::path::db_file;

static MIGRATIONS_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/migrations");

lazy_static! {
    static ref MIGRATIONS: AsyncMigrations =
        AsyncMigrations::from_directory(&MIGRATIONS_DIR).unwrap();
}

#[derive(Clone)]
pub struct DbConn {
    conn: Connection,
}

impl DbConn {
    #[cfg(test)]
    pub async fn new_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().await?;
        DbConn::init_db(conn).await
    }

    pub async fn new() -> Result<Self> {
        tokio::fs::create_dir_all(db_file().parent().unwrap()).await?;
        let conn = Connection::open(db_file()).await?;
        Self::init_db(conn).await
    }

    /// Initialize database, create tables and insert first token if not exist
    async fn init_db(mut conn: Connection) -> Result<Self> {
        MIGRATIONS.to_latest(&mut conn).await?;

        let token = uuid::Uuid::new_v4().to_string();
        conn.call(move |c| {
            Ok(c.execute(
                r#"INSERT OR IGNORE INTO registration_token (id, token) VALUES (1, ?)"#,
                params![token],
            ))
        })
        .await??;

        let res = Self { conn };
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
                Ok(conn.query_row(
                    r#"SELECT token FROM registration_token WHERE id = 1"#,
                    [],
                    |row| row.get(0),
                ))
            })
            .await?;

        Ok(token?)
    }

    /// Update token in database.
    pub async fn reset_registration_token(&self) -> Result<String> {
        let token = uuid::Uuid::new_v4().to_string();
        let result = token.clone();
        let updated_at = chrono::Utc::now();

        let res = self
            .conn
            .call(move |conn| {
                Ok(conn.execute(
                    r#"UPDATE registration_token SET token = ?, updated_at = ? WHERE id = 1"#,
                    params![token, updated_at],
                ))
            })
            .await?;
        if res != Ok(1) {
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
