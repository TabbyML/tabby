use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use anyhow::Result;
use lazy_static::lazy_static;
use rusqlite::params;
use rusqlite_migration::{AsyncMigrations, M};
use tokio_rusqlite::Connection;

lazy_static! {
    static ref MIGRATIONS: AsyncMigrations = AsyncMigrations::new(vec![
        M::up(
            r#"
            CREATE TABLE IF NOT EXISTS register_token (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT (DATETIME('now')),
                updated_at TIMESTAMP DEFAULT (DATETIME('now')),
                CONSTRAINT `idx_token` UNIQUE (`token`)
            );
        "#
        ),
    ]);
}

const TABBY_ROOT: PathBuf = match env::var("TABBY_ROOT") {
    Ok(x) => PathBuf::from(x),
    Err(_) => PathBuf::from(env::var("HOME").unwrap()).join(".tabby"),
};

fn db_file() -> PathBuf {
    TABBY_ROOT.join("db.sqlite3")
}

pub struct DbConn {
    conn: Arc<Connection>,
}

impl DbConn {
    pub async fn new() -> Result<Self> {
        let conn = Connection::open(db_file()).await?;
        Self::init_db(conn).await
    }

    async fn new_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().await?;
        Self::init_db(conn).await
    }

    /// Initialize database, create tables and insert first token.
    async fn init_db(mut conn: Connection) -> Result<Self> {
        MIGRATIONS.to_latest(&mut conn).await?;

        let token = uuid::Uuid::new_v4().to_string();
        let res = conn
            .call(move |c| {
                c.execute(
                    r#"INSERT OR IGNORE INTO token_tab (id, token) VALUES (1, ?)"#,
                    params![token],
                )
            })
            .await?;
        if res != 1 {
            return Err(anyhow::anyhow!("failed to init token"));
        }

        Ok(Self {
            conn: Arc::new(conn),
        })
    }

    /// Query token from database.
    /// Since token is global unique for each tabby server, by right there's only one row in the table.
    pub async fn query_token(&self) -> Result<String> {
        let token = self
            .conn
            .call(|conn| {
                conn.query_row(r#"SELECT token FROM token_tab WHERE id = 1"#, [], |row| {
                    row.get(0)
                })
            })
            .await?;

        Ok(token)
    }

    /// Update token in database.
    pub async fn update_token(&self, token: String) -> Result<()> {
        if token.is_empty() {
            return Err(anyhow::anyhow!("failed: new token is empty"));
        }
        let updated_at = chrono::Utc::now().timestamp() as u32;
        let res = self
            .conn
            .call(move |conn| {
                conn.execute(
                    r#"UPDATE token_tab SET token = ?, updated_at = ? WHERE id = 1"#,
                    params![token, updated_at],
                )
            })
            .await?;
        if res != 1 {
            return Err(anyhow::anyhow!("failed to update token"));
        }
        Ok(())
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
        let token = conn.query_token().await.unwrap();
        assert_eq!(token, "");
    }

    #[tokio::test]
    async fn test_update_token() {
        let conn = DbConn::new_in_memory().await.unwrap();

        // first update
        let new1 = "new_token_1".to_string();
        conn.update_token(new1.clone())
            .await
            .unwrap();
        let token = conn.query_token().await.unwrap();
        assert_eq!(token, new1);

        // second update
        let new2 = "new_token_2".to_string();
        conn.update_token(new1.to_string()).await.unwrap();
        let token = conn.query_token().await.unwrap();
        assert_eq!(token, new2);

        // error case
        let res = conn
            .update_token("".to_string())
            .await;
        assert!(res.is_err());
    }
}
