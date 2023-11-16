use anyhow::Result;
use lazy_static::lazy_static;
use rusqlite_migration::{AsyncMigrations, M};
use tabby_common::path::db_file;
use tokio_rusqlite::Connection;

lazy_static! {
    static ref MIGRATIONS: AsyncMigrations = AsyncMigrations::new(vec![
        M::up(
            r#"
            CREATE TABLE IF NOT EXISTS token_tab (
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
            INSERT OR IGNORE INTO token_tab (id, token) VALUES (1, '');
        "#
        ),
    ]);
}

/// Initialize the database, return a connection.
pub async fn init_db() -> Result<Connection> {
    let mut conn = Connection::open(db_file()).await?;
    MIGRATIONS.to_latest(&mut conn).await?;

    Ok(conn)
}

/// Only used for unit tests.
pub async fn init_memory_db() -> Connection {
    let mut conn = Connection::open_in_memory().await.unwrap();
    MIGRATIONS.to_latest(&mut conn).await.unwrap();

    conn
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn migrations_test() {
        assert!(MIGRATIONS.validate().await.is_ok());
    }
}
