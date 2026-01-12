use std::ops::RangeBounds;

use sqlx::{
    migrate,
    migrate::{Migration, MigrationType, Migrator},
};

use crate::DbConn;

fn migrations(
    migrator: &Migrator,
    range: impl RangeBounds<i64> + 'static,
) -> impl Iterator<Item = &Migration> {
    migrator.iter().filter(move |migration| {
        range.contains(&migration.version)
            && matches!(
                migration.migration_type,
                MigrationType::Simple | MigrationType::ReversibleUp
            )
    })
}

#[tokio::test]
async fn test_repository_migration_0029() {
    let migrator = migrate!("./migrations");
    let db = DbConn::new_blank().await.unwrap();

    for migration in migrations(&migrator, ..29) {
        sqlx::query(&migration.sql).execute(&db.pool).await.unwrap();
    }

    sqlx::query("INSERT INTO github_repository_provider(display_name, access_token) VALUES ('github', 'gh-faketoken');")
            .execute(&db.pool)
            .await
            .unwrap();

    sqlx::query("INSERT INTO github_provided_repositories(github_repository_provider_id, vendor_id, name, git_url, active)
                VALUES (1, 'vendor_id', 'tabby-gh', 'https://github.com/TabbyML/tabby', true);")
            .execute(&db.pool)
            .await
            .unwrap();

    sqlx::query("INSERT INTO gitlab_repository_provider(display_name, access_token) VALUES ('gitlab', 'gl-faketoken');")
            .execute(&db.pool)
            .await
            .unwrap();

    sqlx::query("INSERT INTO gitlab_provided_repositories(gitlab_repository_provider_id, vendor_id, name, git_url, active)
                VALUES (1, 'vendor_id', 'tabby-gl', 'https://gitlab.com/TabbyML/tabby', false);")
            .execute(&db.pool)
            .await
            .unwrap();

    let migration = migrations(&migrator, 29..=29).next().unwrap();
    sqlx::query(&migration.sql).execute(&db.pool).await.unwrap();

    let repos: Vec<(String, i64, String, bool)> = sqlx::query_as(
        "SELECT name, integration_id, git_url, active FROM provided_repositories ORDER BY id",
    )
    .fetch_all(&db.pool)
    .await
    .unwrap();

    assert_eq!(2, repos.len());
    assert_eq!(repos[0].0, "tabby-gh");
    assert_eq!(repos[0].1, 1);
    assert_eq!(repos[0].2, "https://github.com/TabbyML/tabby");
    assert!(repos[0].3);

    assert_eq!(repos[1].0, "tabby-gl");
    assert_eq!(repos[1].1, 2);
    assert_eq!(repos[1].2, "https://gitlab.com/TabbyML/tabby");
    assert!(!repos[1].3);
}
