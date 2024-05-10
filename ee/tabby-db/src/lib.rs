use std::{
    fmt::Display,
    ops::{Add, Deref, Sub},
    path::Path,
    sync::Arc,
};

use anyhow::anyhow;
use cache::Cache;
use cached::TimedSizedCache;
use chrono::{DateTime, Duration, NaiveDateTime, Utc};
pub use email_setting::EmailSettingDAO;
pub use github_repository_provider::{GithubProvidedRepositoryDAO, GithubRepositoryProviderDAO};
pub use gitlab_repository_provider::{GitlabProvidedRepositoryDAO, GitlabRepositoryProviderDAO};
pub use invitations::InvitationDAO;
pub use job_runs::JobRunDAO;
pub use oauth_credential::OAuthCredentialDAO;
pub use repositories::RepositoryDAO;
pub use server_setting::ServerSettingDAO;
use sqlx::{
    database::HasValueRef, query, query_scalar, sqlite::SqliteQueryResult, Decode, Encode, Pool,
    Sqlite, SqlitePool, Type, Value, ValueRef,
};
use tokio::sync::Mutex;
use user_completions::UserCompletionDailyStatsDAO;
pub use user_events::UserEventDAO;
pub use users::UserDAO;

pub mod cache;
mod email_setting;
mod github_repository_provider;
mod gitlab_repository_provider;
mod integration_access_tokens;
mod invitations;
mod job_runs;
mod oauth_credential;
mod password_reset;
mod provided_repositories;
mod refresh_tokens;
mod repositories;
mod server_setting;
mod user_completions;
mod user_events;
mod users;

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

    pub async fn new(db_file: &Path) -> Result<Self> {
        tokio::fs::create_dir_all(db_file.parent().unwrap()).await?;
        let options = SqliteConnectOptions::new()
            .filename(db_file)
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

pub trait DbNullable:
    for<'a> Decode<'a, Sqlite> + for<'a> Encode<'a, Sqlite> + Type<Sqlite>
{
}
impl DbNullable for DateTimeUtc {}

#[derive(Default)]
pub struct DbOption<T>(Option<T>)
where
    T: DbNullable;

impl<T> Type<Sqlite> for DbOption<T>
where
    T: Type<Sqlite> + DbNullable,
{
    fn type_info() -> <Sqlite as sqlx::Database>::TypeInfo {
        T::type_info()
    }
}

impl<'a, T> Decode<'a, Sqlite> for DbOption<T>
where
    T: DbNullable,
{
    fn decode(
        value: <Sqlite as HasValueRef<'a>>::ValueRef,
    ) -> std::prelude::v1::Result<Self, sqlx::error::BoxDynError> {
        if value.is_null() {
            Ok(Self(None))
        } else {
            Ok(Self(Some(T::decode(value)?)))
        }
    }
}

impl<T, F> From<Option<F>> for DbOption<T>
where
    T: From<F> + DbNullable,
{
    fn from(value: Option<F>) -> Self {
        DbOption(value.map(|v| T::from(v)))
    }
}

impl<T> DbOption<T>
where
    T: DbNullable,
{
    pub fn into_option<V>(self) -> Option<V>
    where
        T: Into<V>,
    {
        self.0.map(Into::into)
    }
}

impl<T> Clone for DbOption<T>
where
    T: Clone + DbNullable,
{
    fn clone(&self) -> Self {
        self.0.clone().into()
    }
}

#[derive(Default, Clone)]
pub struct DateTimeUtc(DateTime<Utc>);

impl Display for DateTimeUtc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_sqlite_datetime())
    }
}

impl std::fmt::Debug for DateTimeUtc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_sqlite_datetime())
    }
}

impl Add<Duration> for DateTimeUtc {
    type Output = Self;

    fn add(self, rhs: Duration) -> Self::Output {
        ((self.0) + rhs).into()
    }
}

impl Sub<Duration> for DateTimeUtc {
    type Output = Self;

    fn sub(self, rhs: Duration) -> Self::Output {
        ((self.0) - rhs).into()
    }
}

impl From<DateTime<Utc>> for DateTimeUtc {
    fn from(value: DateTime<Utc>) -> Self {
        Self(value)
    }
}

impl From<DateTimeUtc> for DateTime<Utc> {
    fn from(val: DateTimeUtc) -> Self {
        *val
    }
}

impl<'a> Decode<'a, Sqlite> for DateTimeUtc {
    fn decode(
        value: <Sqlite as HasValueRef<'a>>::ValueRef,
    ) -> std::prelude::v1::Result<Self, sqlx::error::BoxDynError> {
        let time: NaiveDateTime = value.to_owned().decode();
        Ok(time.into())
    }
}

impl Type<Sqlite> for DateTimeUtc {
    fn type_info() -> <Sqlite as sqlx::Database>::TypeInfo {
        <String as Type<Sqlite>>::type_info()
    }
}

impl<'a> Encode<'a, Sqlite> for DateTimeUtc {
    fn encode_by_ref(
        &self,
        buf: &mut <Sqlite as sqlx::database::HasArguments<'a>>::ArgumentBuffer,
    ) -> sqlx::encode::IsNull {
        <String as Encode<Sqlite>>::encode(self.as_sqlite_datetime(), buf)
    }
}

impl From<NaiveDateTime> for DateTimeUtc {
    fn from(value: NaiveDateTime) -> Self {
        DateTimeUtc(value.and_utc())
    }
}

impl Deref for DateTimeUtc {
    type Target = DateTime<Utc>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PartialEq for DateTimeUtc {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for DateTimeUtc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Copy for DateTimeUtc {}

impl DateTimeUtc {
    pub fn now() -> Self {
        Self(Utc::now())
    }

    pub fn from_timestamp(secs: i64, subsec_nanos: u32) -> Option<Self> {
        DateTime::from_timestamp(secs, subsec_nanos).map(Self)
    }

    fn as_sqlite_datetime(&self) -> String {
        self.0.format("%F %X").to_string()
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

    #[tokio::test]
    async fn test_timestamp_format() {
        let db = DbConn::new_in_memory().await.unwrap();

        let time = DateTimeUtc::now();

        let time_str = time.as_sqlite_datetime();
        let sql_time: String = sqlx::query_scalar::<_, String>("SELECT ?;")
            .bind(time)
            .fetch_one(&db.pool)
            .await
            .unwrap();

        assert_eq!(time_str, sql_time);

        let sql_time: String = sqlx::query_scalar::<_, String>("SELECT DATETIME('now');")
            .fetch_one(&db.pool)
            .await
            .unwrap();
        assert_eq!(sql_time, DateTimeUtc::now().as_sqlite_datetime());

        // No assertions, these will fail at compiletime if adding/subtracting from these types
        // yields DateTime<Utc>, which could be dangerous
        let time = DateTimeUtc::now();
        let _added_time: DateTimeUtc = time + Duration::milliseconds(1);
        let _subbed_time: DateTimeUtc = time - Duration::milliseconds(1);
    }
}

#[cfg(any(test, feature = "testutils"))]
pub mod testutils {
    use super::*;

    pub async fn create_user(conn: &DbConn) -> i64 {
        let email: &str = "test@example.com";
        let password: &str = "123456789";
        conn.create_user(email.to_string(), Some(password.to_string()), true)
            .await
            .unwrap()
    }
}
