pub use email_service_credential::EmailServiceCredentialDAO;
pub use github_oauth_credential::GithubOAuthCredentialDAO;
pub use invitations::InvitationDAO;
pub use job_runs::JobRunDAO;
pub use users::UserDAO;

mod email_service_credential;
mod github_oauth_credential;
mod invitations;
mod job_runs;
mod path;
mod refresh_tokens;
mod users;

use anyhow::Result;
use include_dir::{include_dir, Dir};
use lazy_static::lazy_static;
use rusqlite::params;
use rusqlite_migration::AsyncMigrations;
use tokio_rusqlite::Connection;

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
    #[cfg(any(test, feature = "testutils"))]
    pub async fn new_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().await?;
        DbConn::init_db(conn).await
    }

    pub async fn new() -> Result<Self> {
        tokio::fs::create_dir_all(path::db_file().parent().unwrap()).await?;
        let conn = Connection::open(path::db_file()).await?;
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

    fn make_pagination_query(
        table_name: &str,
        field_names: &[&str],
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> String {
        let mut source = String::new();
        let mut clause = String::new();
        if backwards {
            source += &format!("SELECT * FROM {}", table_name);
            if let Some(skip_id) = skip_id {
                source += &format!(" WHERE id < {}", skip_id);
            }
            source += " ORDER BY id DESC";
            if let Some(limit) = limit {
                source += &format!(" LIMIT {}", limit);
            }
            clause += " ORDER BY id ASC";
        } else {
            source += table_name;
            if let Some(skip_id) = skip_id {
                clause += &format!(" WHERE id > {}", skip_id);
            }
            clause += " ORDER BY id ASC";
            if let Some(limit) = limit {
                clause += &format!(" LIMIT {}", limit);
            }
        }
        let fields = field_names.join(", ");

        format!(r#"SELECT {} FROM ({}) {}"#, fields, source, clause)
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
}

#[cfg(any(test, feature = "testutils"))]
pub mod testutils {
    use super::*;

    pub async fn create_user(conn: &DbConn) -> i32 {
        let email: &str = "test@example.com";
        let password: &str = "123456789";
        conn.create_user(email.to_string(), password.to_string(), true)
            .await
            .unwrap()
    }
}
