use std::{path::PathBuf, sync::Arc};

use anyhow::{anyhow, Result};
use lazy_static::lazy_static;
use rusqlite::{params, OptionalExtension, Row};
use rusqlite_migration::{AsyncMigrations, M};
use tabby_common::path::tabby_root;
use tokio_rusqlite::Connection;
use uuid::Uuid;

use crate::{schema::auth::Invitation, service::cron::run_offline_job};

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
                expires_at INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT (DATETIME('now')),
                CONSTRAINT `idx_token` UNIQUE (`token`)
            );
        "#
        )
        .down("DROP TABLE refresh_tokens"),
    ]);
}

#[allow(unused)]
pub struct User {
    created_at: String,
    updated_at: String,

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

/// db read/write operations for `users` table
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

impl Invitation {
    fn from_row(row: &Row<'_>) -> std::result::Result<Self, rusqlite::Error> {
        Ok(Self {
            id: row.get(0)?,
            email: row.get(1)?,
            code: row.get(2)?,
            created_at: row.get(3)?,
        })
    }
}

/// db read/write operations for `invitations` table
impl DbConn {
    pub async fn list_invitations(&self) -> Result<Vec<Invitation>> {
        let invitations = self
            .conn
            .call(move |c| {
                let mut stmt =
                    c.prepare(r#"SELECT id, email, code, created_at FROM invitations"#)?;
                let iter = stmt.query_map([], Invitation::from_row)?;
                Ok(iter.filter_map(|x| x.ok()).collect::<Vec<_>>())
            })
            .await?;

        Ok(invitations)
    }

    pub async fn get_invitation_by_code(&self, code: &str) -> Result<Option<Invitation>> {
        let code = code.to_owned();
        let token = self
            .conn
            .call(|conn| {
                conn.query_row(
                    r#"SELECT id, email, code, created_at FROM invitations WHERE code = ?"#,
                    [code],
                    Invitation::from_row,
                )
                .optional()
            })
            .await?;

        Ok(token)
    }

    pub async fn create_invitation(&self, email: String) -> Result<i32> {
        let code = Uuid::new_v4().to_string();
        let res = self
            .conn
            .call(move |c| {
                let mut stmt =
                    c.prepare(r#"INSERT INTO invitations (email, code) VALUES (?, ?)"#)?;
                let rowid = stmt.insert((email, code))?;
                Ok(rowid)
            })
            .await?;
        if res != 1 {
            return Err(anyhow!("failed to create invitation"));
        }

        Ok(res as i32)
    }

    pub async fn delete_invitation(&self, id: i32) -> Result<i32> {
        let res = self
            .conn
            .call(move |c| c.execute(r#"DELETE FROM invitations WHERE id = ?"#, params![id]))
            .await?;
        if res != 1 {
            return Err(anyhow!("failed to delete invitation"));
        }

        Ok(id)
    }
}

#[allow(unused)]
pub struct RefreshToken {
    id: u32,
    created_at: String,

    pub user_id: i32,
    pub token: String,
    pub expires_at: i64,
}

impl RefreshToken {
    fn select(clause: &str) -> String {
        r#"SELECT id, user_id, token, expires_at, created_at FROM refresh_tokens WHERE "#.to_owned()
            + clause
    }

    fn from_row(row: &Row<'_>) -> std::result::Result<RefreshToken, rusqlite::Error> {
        Ok(RefreshToken {
            id: row.get(0)?,
            user_id: row.get(1)?,
            token: row.get(2)?,
            expires_at: row.get(3)?,
            created_at: row.get(4)?,
        })
    }

    pub fn is_expired(&self) -> bool {
        let now = chrono::Utc::now().timestamp();
        self.expires_at < now
    }
}

/// db read/write operations for `refresh_tokens` table
impl DbConn {
    pub async fn create_refresh_token(
        &self,
        user_id: i32,
        token: &str,
        expires_at: i64,
    ) -> Result<()> {
        let token = token.to_string();
        let res = self
            .conn
            .call(move |c| {
                c.execute(
                    r#"INSERT INTO refresh_tokens (user_id, token, expires_at) VALUES (?, ?, ?)"#,
                    params![user_id, token, expires_at],
                )
            })
            .await?;
        if res != 1 {
            return Err(anyhow::anyhow!("failed to create refresh token"));
        }

        Ok(())
    }

    pub async fn replace_refresh_token(&self, old: &str, new: &str) -> Result<()> {
        let old = old.to_string();
        let new = new.to_string();
        let res = self
            .conn
            .call(move |c| {
                c.execute(
                    r#"UPDATE refresh_tokens SET token = ? WHERE token = ?"#,
                    params![new, old],
                )
            })
            .await?;
        if res != 1 {
            return Err(anyhow::anyhow!("failed to replace refresh token"));
        }

        Ok(())
    }

    pub async fn delete_expired_token(&self, utc_ts: i64) -> Result<i32> {
        let res = self
            .conn
            .call(move |c| {
                c.execute(
                    r#"DELETE FROM refresh_tokens WHERE expires_at < ?"#,
                    params![utc_ts],
                )
            })
            .await?;

        Ok(res as i32)
    }

    pub async fn get_refresh_token(&self, token: &str) -> Result<Option<RefreshToken>> {
        let token = token.to_string();
        let token = self
            .conn
            .call(move |c| {
                c.query_row(
                    RefreshToken::select("token = ?").as_str(),
                    params![token],
                    RefreshToken::from_row,
                )
                .optional()
            })
            .await?;

        Ok(token)
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
    async fn test_is_admin_initialized() {
        let conn = DbConn::new_in_memory().await.unwrap();

        assert!(!conn.is_admin_initialized().await.unwrap());
        create_user(&conn).await;
        assert!(conn.is_admin_initialized().await.unwrap());
    }

    #[tokio::test]
    async fn test_invitations() {
        let conn = DbConn::new_in_memory().await.unwrap();

        let email = "hello@example.com".to_owned();
        conn.create_invitation(email).await.unwrap();

        let invitations = conn.list_invitations().await.unwrap();
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

        let invitations = conn.list_invitations().await.unwrap();
        assert!(invitations.is_empty());
    }

    #[tokio::test]
    async fn test_create_refresh_token() {
        let conn = DbConn::new_in_memory().await.unwrap();

        conn.create_refresh_token(1, "test", 100).await.unwrap();

        let token = conn.get_refresh_token("test").await.unwrap().unwrap();

        assert_eq!(token.user_id, 1);
        assert_eq!(token.token, "test");
        assert_eq!(token.expires_at, 100);
    }

    #[tokio::test]
    async fn test_replace_refresh_token() {
        let conn = DbConn::new_in_memory().await.unwrap();

        conn.create_refresh_token(1, "test", 100).await.unwrap();
        conn.replace_refresh_token("test", "test2").await.unwrap();

        let token = conn.get_refresh_token("test").await.unwrap();
        assert!(token.is_none());

        let token = conn.get_refresh_token("test2").await.unwrap().unwrap();
        assert_eq!(token.user_id, 1);
        assert_eq!(token.token, "test2");
        assert_eq!(token.expires_at, 100);
    }
}
