use std::{ops::Deref, sync::Arc};

use anyhow::anyhow;
use cache::Cache;
use chrono::{DateTime, NaiveDateTime, Utc};
pub use email_setting::EmailSettingDAO;
pub use github_oauth_credential::GithubOAuthCredentialDAO;
pub use google_oauth_credential::GoogleOAuthCredentialDAO;
pub use invitations::InvitationDAO;
pub use job_runs::JobRunDAO;
pub use repositories::RepositoryDAO;
pub use server_setting::ServerSettingDAO;
use sqlx::{query, query_scalar, sqlite::SqliteQueryResult, Pool, Sqlite, SqlitePool};
pub use users::UserDAO;

pub mod cache;
mod email_setting;
mod github_oauth_credential;
mod google_oauth_credential;
mod invitations;
mod job_runs;
mod password_reset;
mod path;
mod refresh_tokens;
mod repositories;
mod server_setting;
mod users;

use anyhow::Result;
use sql_query_builder as sql;
use sqlx::sqlite::SqliteConnectOptions;

pub trait DbEnum: Sized {
    fn as_enum_str(&self) -> &'static str;
    fn from_enum_str(s: &str) -> anyhow::Result<Self>;
}

#[derive(Default)]
pub struct DbCache {
    pub active_user_count: Cache<usize>,
    pub active_admin_count: Cache<usize>,
}

#[derive(Clone)]
pub struct DbConn {
    pool: Pool<Sqlite>,
    cache: Arc<DbCache>,
}

impl DbConn {
    #[cfg(any(test, feature = "testutils"))]
    pub async fn new_in_memory() -> Result<Self> {
        use std::str::FromStr;

        use sqlx::sqlite::SqlitePoolOptions;

        let options = SqliteConnectOptions::from_str("sqlite::memory:")?;
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await?;
        DbConn::init_db(pool).await
    }

    pub async fn new() -> Result<Self> {
        tokio::fs::create_dir_all(path::db_file().parent().unwrap()).await?;
        let options = SqliteConnectOptions::new()
            .filename(path::db_file())
            .create_if_missing(true);
        let pool = SqlitePool::connect_with(options).await?;
        Self::init_db(pool).await
    }

    /// Initialize database, create tables and insert first token if not exist
    async fn init_db(pool: SqlitePool) -> Result<Self> {
        sqlx::migrate!("./migrations").run(&pool).await?;

        let token = uuid::Uuid::new_v4().to_string();
        query!(
            "INSERT OR IGNORE INTO registration_token (id, token) VALUES (1, ?)",
            token
        )
        .execute(&pool)
        .await?;

        let conn = Self {
            pool,
            cache: Default::default(),
        };
        conn.manual_users_active_migration().await?;
        Ok(conn)
    }

    /// This migration is applied manually to make the transition between rusqlite and sqlx smooth,
    /// since there is no way to conditionally alter a table with a pure SQLite script.
    /// Once all users can reasonably be expected to have moved to the sqlx version,
    /// we can remove this function.
    async fn manual_users_active_migration(&self) -> Result<()> {
        let active_exists =
            sqlx::query("SELECT * FROM pragma_table_info('users') WHERE name='active'")
                .fetch_optional(&self.pool)
                .await?;

        if active_exists.is_none() {
            sqlx::query("ALTER TABLE users ADD COLUMN active BOOLEAN NOT NULL DEFAULT 1")
                .execute(&self.pool)
                .await?;
        }
        Ok(())
    }

    fn make_pagination_query_with_condition(
        table_name: &str,
        field_names: &[&str],
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
        condition: Option<String>,
    ) -> String {
        let mut source = sql::Select::new().select("*").from(table_name);
        let mut select = sql::Select::new()
            .select(&field_names.join(", "))
            .order_by("id ASC");

        if backwards {
            source = source.order_by("id DESC");
            if let Some(skip_id) = skip_id {
                source = source.where_and(&format!("id < {skip_id}"));
            }
            if let Some(limit) = limit {
                source = source.limit(&limit.to_string());
            }
        } else {
            if let Some(skip_id) = skip_id {
                select = select.where_and(&format!("id > {skip_id}"));
            }
            if let Some(limit) = limit {
                select = select.limit(&limit.to_string());
            }
        }

        select = select.from(&format!("({source})"));
        if let Some(condition) = condition {
            select = select.where_and(&condition)
        }

        select.as_string()
    }

    fn make_pagination_query(
        table_name: &str,
        field_names: &[&str],
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> String {
        Self::make_pagination_query_with_condition(
            table_name,
            field_names,
            limit,
            skip_id,
            backwards,
            None,
        )
    }
}

/// db read/write operations for `registration_token` table
impl DbConn {
    /// Query token from database.
    /// Since token is global unique for each tabby server, by right there's only one row in the table.
    pub async fn read_registration_token(&self) -> Result<String> {
        Ok(
            query_scalar!("SELECT token FROM registration_token WHERE id = 1")
                .fetch_one(&self.pool)
                .await?,
        )
    }

    /// Update token in database.
    pub async fn reset_registration_token(&self) -> Result<String> {
        let token = uuid::Uuid::new_v4().to_string();
        let result = token.clone();
        let updated_at = chrono::Utc::now();

        let res = query!(
            "UPDATE registration_token SET token = ?, updated_at = ? WHERE id = 1",
            token,
            updated_at
        )
        .execute(&self.pool)
        .await?;
        if res.rows_affected() != 1 {
            return Err(anyhow::anyhow!("failed to update token"));
        }

        Ok(result)
    }
}

pub struct DateTimeUtc(DateTime<Utc>);

impl From<NaiveDateTime> for DateTimeUtc {
    fn from(value: NaiveDateTime) -> Self {
        let utc = DateTime::from_naive_utc_and_offset(value, Utc);
        DateTimeUtc(utc)
    }
}

impl Deref for DateTimeUtc {
    type Target = DateTime<Utc>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DateTimeUtc {
    pub fn into_inner(self) -> DateTime<Utc> {
        self.0
    }
}

pub trait SQLXResultExt {
    fn unique_error(self, msg: &'static str) -> anyhow::Result<SqliteQueryResult>;
}

impl SQLXResultExt for Result<SqliteQueryResult, sqlx::Error> {
    fn unique_error(self, msg: &'static str) -> anyhow::Result<SqliteQueryResult> {
        match self {
            Ok(v) => Ok(v),
            Err(sqlx::Error::Database(db_err)) if db_err.is_unique_violation() => {
                Err(anyhow!("{msg}"))
            }
            Err(e) => Err(e.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
