use std::path::Path;

use sqlx::migrate::MigrateError;
use sqlx_migrate_validate::{Validate, ValidateError, Validator};

async fn prepare() -> sqlx::sqlite::SqliteConnection {
    let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
    let mut conn = pool.acquire().await.unwrap();

    sqlx::migrate!("./tests/migrations-1")
        .run(&mut conn)
        .await
        .unwrap();

    conn.detach()
}

#[tokio::test]
async fn validate_returns_ok_with_same_migrations() {
    let mut conn = prepare().await;

    sqlx::migrate!("./tests/migrations-1")
        .validate(&mut conn)
        .await
        .unwrap();
}

#[tokio::test]
async fn validate_returns_err_with_more_migrations_in_source() {
    let mut conn = prepare().await;

    match Validator::new(Path::new("./tests/migrations-2"))
        .await
        .unwrap()
        .validate(&mut conn)
        .await
    {
        Err(ValidateError::VersionNotApplied(20230312141719)) => (),
        o => panic!("Expected VersionNotApplied error, got: {o:?}"),
    };
}

#[tokio::test]
async fn validate_returns_err_with_less_migrations_in_source() {
    let mut conn = prepare().await;

    match Validator::new(Path::new("./tests/migrations-3"))
        .await
        .unwrap()
        .validate(&mut conn)
        .await
    {
        Err(ValidateError::MigrateError(MigrateError::VersionMissing(20230312133715))) => (),
        o => panic!("Expected VersionMissing error, got: {o:?}"),
    };
}

#[tokio::test]
async fn validate_returns_err_with_migration_content_mismatch() {
    let mut conn = prepare().await;

    match Validator::new(Path::new("./tests/migrations-4"))
        .await
        .unwrap()
        .validate(&mut conn)
        .await
    {
        Err(ValidateError::MigrateError(MigrateError::VersionMismatch(20230312133715))) => (),
        o => panic!("Expected VersionMismatch error, got: {o:?}"),
    };
}
