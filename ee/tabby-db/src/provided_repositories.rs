use anyhow::{anyhow, Result};
use sqlx::{prelude::FromRow, query, query_as};
use tabby_db_macros::query_paged_as;

use crate::{DateTimeUtc, DbConn};

#[derive(FromRow)]
pub struct ProvidedRepositoryDAO {
    pub id: i64,
    pub vendor_id: String,
    pub integration_id: i64,
    pub name: String,
    pub git_url: String,
    pub active: bool,
    pub created_at: DateTimeUtc,
    pub updated_at: DateTimeUtc,
}

impl DbConn {
    pub async fn upsert_provided_repository(
        &self,
        integration_id: i64,
        vendor_id: String,
        name: String,
        git_url: String,
    ) -> Result<i64> {
        let res = query!(
            "INSERT INTO provided_repositories (integration_id, vendor_id, name, git_url) VALUES ($1, $2, $3, $4)
                ON CONFLICT(integration_id, vendor_id) DO UPDATE SET name = $3, git_url = $4, updated_at = DATETIME('now')",
            integration_id,
            vendor_id,
            name,
            git_url
        ).execute(&self.pool).await?;
        Ok(res.last_insert_rowid())
    }

    pub async fn delete_outdated_provided_repositories(
        &self,
        integration_id: i64,
        cutoff_timestamp: DateTimeUtc,
    ) -> Result<usize> {
        let res = query!(
            "DELETE FROM provided_repositories WHERE integration_id = ? AND updated_at < ?;",
            integration_id,
            cutoff_timestamp
        )
        .execute(&self.pool)
        .await?;
        Ok(res.rows_affected() as usize)
    }

    pub async fn get_provided_repository(&self, id: i64) -> Result<ProvidedRepositoryDAO> {
        let repo = query_as!(
            ProvidedRepositoryDAO,
            "SELECT id, vendor_id, name, git_url, active, integration_id, created_at, updated_at FROM provided_repositories WHERE id = ?",
            id
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(repo)
    }

    pub async fn list_provided_repositories(
        &self,
        integration_ids: Vec<i64>,
        kind: Option<String>,
        active: Option<bool>,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<ProvidedRepositoryDAO>> {
        let mut conditions = vec![];

        let integration_ids = integration_ids
            .into_iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        if !integration_ids.is_empty() {
            conditions.push(format!("integration_id IN ({integration_ids})"));
        }

        let active_filter = active.map(|active| format!("active = {active}"));
        conditions.extend(active_filter);

        let kind_filter = kind.map(|kind| format!("kind = '{kind}'"));
        conditions.extend(kind_filter);

        let condition = (!conditions.is_empty()).then(|| conditions.join(" AND "));

        let repos = query_paged_as!(
            ProvidedRepositoryDAO,
            "provided_repositories JOIN integrations ON integration_id = integrations.id",
            [
                "id",
                "vendor_id",
                "name",
                "git_url",
                "active",
                "integration_id",
                "created_at",
                "updated_at"
            ],
            limit,
            skip_id,
            backwards,
            condition
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(repos)
    }

    pub async fn update_provided_repository_active(&self, id: i64, active: bool) -> Result<()> {
        let not_active = !active;
        let res = query!(
            "UPDATE provided_repositories SET active = ? WHERE id = ? AND active = ?",
            active,
            id,
            not_active
        )
        .execute(&self.pool)
        .await?;

        if res.rows_affected() != 1 {
            return Err(anyhow!("Repository active status was not changed"));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use sqlx::migrate;

    use crate::DbConn;

    #[tokio::test]
    async fn test_list_provided_repositories() {
        let db = DbConn::new_in_memory().await.unwrap();
        // Ensure query does not break on the join
        db.list_provided_repositories(vec![], Some("github".into()), None, None, None, false)
            .await
            .unwrap();
    }

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
}
