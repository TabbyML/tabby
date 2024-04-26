use anyhow::{anyhow, Result};
use sqlx::{prelude::FromRow, query};
use tabby_db_macros::query_paged_as;

use crate::{DbConn, SQLXResultExt};

#[derive(FromRow)]
pub struct RepositoryDAO {
    pub id: i64,
    pub name: String,
    pub git_url: String,
}

impl DbConn {
    pub async fn list_repositories_with_filter(
        &self,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<RepositoryDAO>> {
        let repos = query_paged_as!(
            RepositoryDAO,
            "repositories",
            ["id", "name", "git_url"],
            limit,
            skip_id,
            backwards
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(repos)
    }

    pub async fn delete_repository(&self, id: i64) -> Result<bool> {
        let res = query!("DELETE FROM repositories WHERE id = ?", id)
            .execute(&self.pool)
            .await?;
        Ok(res.rows_affected() == 1)
    }

    pub async fn create_repository(&self, name: String, git_url: String) -> Result<i64> {
        let res = query!(
            "INSERT INTO repositories (name, git_url) VALUES (?, ?)",
            name,
            git_url
        )
        .execute(&self.pool)
        .await;

        res.unique_error("A repository with the same name or URL already exists")
            .map(|output| output.last_insert_rowid())
    }

    pub async fn update_repository(&self, id: i64, name: String, git_url: String) -> Result<()> {
        let rows = query!(
            "UPDATE repositories SET name = ?, git_url = ? WHERE id = ?",
            name,
            git_url,
            id
        )
        .execute(&self.pool)
        .await?;
        if rows.rows_affected() == 1 {
            Ok(())
        } else {
            Err(anyhow!("failed to update: repository not found"))
        }
    }

    pub async fn get_repository(&self, id: i64) -> Result<RepositoryDAO> {
        let repository = sqlx::query_as!(
            RepositoryDAO,
            "SELECT id as 'id!: i64', name, git_url FROM repositories WHERE id = ?",
            id
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(repository)
    }
}

#[cfg(test)]
mod tests {
    use crate::DbConn;

    #[tokio::test]
    async fn test_update_repository() {
        let conn = DbConn::new_in_memory().await.unwrap();

        // Insert new repository
        conn.create_repository("test".into(), "testurl".into())
            .await
            .unwrap();

        // Test that the url can be retrieved
        let repository = &conn
            .list_repositories_with_filter(None, None, false)
            .await
            .unwrap()[0];
        assert_eq!(repository.git_url, "testurl");

        // Update the repository
        let id = repository.id;
        conn.update_repository(id, "test2".into(), "testurl2".into())
            .await
            .unwrap();

        // Check the url was updated
        let repository = &conn
            .list_repositories_with_filter(None, None, false)
            .await
            .unwrap()[0];
        assert_eq!(repository.git_url, "testurl2");
        assert_eq!(repository.name, "test2");
    }
}
