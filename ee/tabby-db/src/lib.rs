use std::{path::Path, sync::Arc};

use anyhow::anyhow;
pub use attachment::{
    Attachment, AttachmentClientCode, AttachmentCode, AttachmentCodeFileList, AttachmentCommit,
    AttachmentDoc, AttachmentIssueDoc, AttachmentPullDoc, AttachmentWebDoc,
};
use cache::Cache;
use cached::TimedSizedCache;
use chrono::{DateTime, Utc};
pub use email_setting::EmailSettingDAO;
pub use integrations::IntegrationDAO;
pub use invitations::InvitationDAO;
pub use job_runs::JobRunDAO;
pub use ldap_credential::LdapCredentialDAO;
pub use notifications::NotificationDAO;
pub use oauth_credential::OAuthCredentialDAO;
pub use pages::{PageDAO, PageSectionDAO};
pub use provided_repositories::ProvidedRepositoryDAO;
pub use repositories::RepositoryDAO;
pub use server_setting::ServerSettingDAO;
use sqlx::{query, query_scalar, sqlite::SqliteQueryResult, Pool, Sqlite, SqlitePool};
pub use threads::{ThreadDAO, ThreadMessageDAO};
use tokio::sync::Mutex;
use user_completions::UserCompletionDailyStatsDAO;
pub use user_events::UserEventDAO;
pub use user_groups::{UserGroupDAO, UserGroupMembershipDAO};
pub use users::UserDAO;
pub use web_documents::WebDocumentDAO;

mod access_policy;
mod attachment;
pub mod cache;
mod email_setting;
mod integrations;
mod invitations;
mod job_runs;
mod ldap_credential;
#[cfg(test)]
mod migration_tests;
mod notifications;
mod oauth_credential;
mod pages;
mod password_reset;
mod provided_repositories;
mod refresh_tokens;
mod repositories;
mod server_setting;
mod threads;
mod user_completions;
mod user_events;
mod user_groups;
mod users;
mod web_documents;

use anyhow::Result;
use sql_query_builder as sql;
use sqlx::sqlite::SqliteConnectOptions;

pub struct DbCache {
    pub active_user_count: Cache<usize>,
    pub active_admin_count: Cache<usize>,
    pub daily_stats_in_past_year:
        Arc<Mutex<TimedSizedCache<Option<i64>, Vec<UserCompletionDailyStatsDAO>>>>,
}

#[derive(Clone)]
pub struct DbConn {
    pool: Pool<Sqlite>,
    cache: Arc<DbCache>,
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

    if let Some(condition) = condition {
        source = source.where_and(&condition)
    }

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
    select.as_string()
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

    #[cfg(any(test, feature = "testutils"))]
    pub async fn new_blank() -> Result<Self> {
        use std::str::FromStr;

        use sqlx::sqlite::SqlitePoolOptions;

        let options = SqliteConnectOptions::from_str("sqlite::memory:")?;
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await?;
        Ok(DbConn {
            pool,
            cache: Arc::new(DbCache {
                active_user_count: Default::default(),
                active_admin_count: Default::default(),
                daily_stats_in_past_year: Arc::new(Mutex::new(
                    TimedSizedCache::with_size_and_lifespan(20, 3600),
                )),
            }),
        })
    }

    /// We forked sqlx to disable support for chrono::DateTime<Utc> as it's format is problematic
    /// against SQLite `DATETIME("now")`.
    ///
    /// ```compile_fail
    /// let output = sqlx::query_scalar::<_, String>("SELECT ?;").bind(chrono::Utc::now());
    /// ```
    fn _datetime_utc_shouldnt_be_bindable() {}

    pub async fn new(db_file: &Path) -> Result<Self> {
        tokio::fs::create_dir_all(db_file.parent().unwrap()).await?;

        let options = SqliteConnectOptions::new()
            // Reduce SQLITE_BUSY (code 5) errors. Note that the error message "database is locked" should not be confused with SQLITE_LOCKED.
            // For more details, see:
            // 1. https://til.simonwillison.net/sqlite/enabling-wal-mode
            // 2. https://www.sqlite.org/wal.html
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
            .filename(db_file)
            .create_if_missing(true);
        let pool = SqlitePool::connect_with(options).await?;
        Self::backup_db(db_file, &pool).await?;
        Self::init_db(pool).await
    }

    /// Backup existing database file before opening it.
    /// backup format:
    /// for prod - db.backup-${date}.sqlite
    /// for non-prod - dev-db.backup-${date}.sqlite
    async fn backup_db(db_file: &Path, pool: &SqlitePool) -> Result<()> {
        use sqlx_migrate_validate::Validate;

        let mut conn = pool.acquire().await?;
        if sqlx::migrate!("./migrations")
            .validate(&mut *conn)
            .await
            .is_ok()
        {
            // No migration is needed, skip the backup.
            return Ok(());
        }

        if !tokio::fs::try_exists(db_file).await? {
            return Ok(());
        }
        let Some(db_file_name) = db_file.file_name() else {
            return Err(anyhow!("failed to backup db, missing db file name"));
        };
        let db_file_name = db_file_name.to_string_lossy();
        if !db_file_name.ends_with(".sqlite") {
            return Err(anyhow!("failed to backup db, expect .sqlite extension"));
        }

        let today = Utc::now().date_naive().format("%Y%m%d").to_string();
        let backup_file = db_file.with_file_name(
            db_file_name.replace(".sqlite", format!(".backup-{}.sqlite", today).as_str()),
        );

        tokio::fs::copy(db_file, &backup_file).await?;
        Ok(())
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
            cache: Arc::new(DbCache {
                active_user_count: Default::default(),
                active_admin_count: Default::default(),
                daily_stats_in_past_year: Arc::new(Mutex::new(
                    TimedSizedCache::with_size_and_lifespan(20, 3600),
                )),
            }),
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
        let updated_at = Utc::now().as_sqlite_datetime();

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

trait AsSqliteDateTimeString {
    fn as_sqlite_datetime(&self) -> String;
}

impl AsSqliteDateTimeString for DateTime<Utc> {
    fn as_sqlite_datetime(&self) -> String {
        self.format("%F %X").to_string()
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
    use chrono::Duration;

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

    #[tokio::test]
    async fn test_timestamp_format() {
        let db = DbConn::new_in_memory().await.unwrap();

        let time = Utc::now();

        let time_str = time.as_sqlite_datetime();
        let sql_time: String = sqlx::query_scalar::<_, String>("SELECT ?;")
            .bind(time.as_sqlite_datetime())
            .fetch_one(&db.pool)
            .await
            .unwrap();

        assert_eq!(time_str, sql_time);

        let sql_time: String = sqlx::query_scalar::<_, String>("SELECT DATETIME('now');")
            .fetch_one(&db.pool)
            .await
            .unwrap();
        assert_eq!(sql_time, Utc::now().as_sqlite_datetime());

        // No assertions, these will fail at compiletime if adding/subtracting from these types
        // yields DateTime<Utc>, which could be dangerous
        let time = Utc::now();
        let _added_time: DateTime<Utc> = time + Duration::milliseconds(1);
        let _subbed_time: DateTime<Utc> = time - Duration::milliseconds(1);
    }
}

#[cfg(any(test, feature = "testutils"))]
pub mod testutils {
    use super::*;

    pub async fn create_user(conn: &DbConn) -> i64 {
        let email: &str = "test@example.com";
        let password: &str = "123456789";
        conn.create_user(email.to_string(), Some(password.to_string()), true, None)
            .await
            .unwrap()
    }

    pub async fn create_user2(conn: &DbConn) -> i64 {
        let email: &str = "test2@example.com";
        let password: &str = "123456789";
        conn.create_user(email.to_string(), Some(password.to_string()), true, None)
            .await
            .unwrap()
    }

    pub async fn create_user3(conn: &DbConn) -> i64 {
        let email: &str = "test3@example.com";
        let password: &str = "123456789";
        conn.create_user(email.to_string(), Some(password.to_string()), true, None)
            .await
            .unwrap()
    }
}
