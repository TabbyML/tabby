use sqlx::migrate;

use crate::DbConn;

#[tokio::test]
async fn test_repository_migration() {
    let migrations = migrate!("./migrations");
    let db = DbConn::new_in_memory().await.unwrap();

    migrations.undo(&db.pool, 28).await.unwrap();

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

    migrations.run(&db.pool).await.unwrap();

    let repos = db
        .list_provided_repositories(vec![], None, None, None, None, false)
        .await
        .unwrap();

    assert_eq!(2, repos.len());
    assert_eq!(repos[0].name, "tabby-gh");
    assert_eq!(repos[0].integration_id, 1);
    assert_eq!(repos[0].git_url, "https://github.com/TabbyML/tabby");
    assert!(repos[0].active);

    assert_eq!(repos[1].name, "tabby-gl");
    assert_eq!(repos[1].integration_id, 2);
    assert_eq!(repos[1].git_url, "https://gitlab.com/TabbyML/tabby");
    assert!(!repos[1].active);
}
