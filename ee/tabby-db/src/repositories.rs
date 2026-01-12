use anyhow::{anyhow, Result};
use sqlx::{prelude::FromRow, query};
use tabby_db_macros::query_paged_as;

use crate::{DbConn, SQLXResultExt};

#[derive(FromRow)]
pub struct RepositoryDAO {
    pub id: i64,
    pub name: String,
    pub git_url: String,
    pub refs: Option<String>,
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
            ["id", "name", "git_url", "refs"],
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

    pub async fn create_repository(
        &self,
        name: String,
        git_url: String,
        refs: Vec<String>,
    ) -> Result<i64> {
        let refs = if refs.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&refs)?)
        };

        let res = query!(
            "INSERT INTO repositories (name, git_url, refs) VALUES (?, ?, ?)",
            name,
            git_url,
            refs
        )
        .execute(&self.pool)
        .await;

        res.unique_error("A repository with the same name or URL already exists")
            .map(|output| output.last_insert_rowid())
    }

    pub async fn update_repository(
        &self,
        id: i64,
        name: String,
        git_url: String,
        refs: Vec<String>,
    ) -> Result<()> {
        let refs = if refs.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&refs)?)
        };

        let rows = query!(
            "UPDATE repositories SET name = ?, git_url = ?, refs = ? WHERE id = ?",
            name,
            git_url,
            refs,
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
            "SELECT id as 'id!: i64', name, git_url, refs FROM repositories WHERE id = ?",
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
        conn.create_repository("test".into(), "testurl".into(), vec!["main".into()])
            .await
            .unwrap();

        // Test that the url can be retrieved
        let repository = &conn
            .list_repositories_with_filter(None, None, false)
            .await
            .unwrap()[0];
        assert_eq!(repository.git_url, "testurl");
        assert_eq!(repository.refs, Some("[\"main\"]".to_string()));

        // Update the repository
        let id = repository.id;
        conn.update_repository(id, "test2".into(), "testurl2".into(), vec!["main".into()])
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

    #[tokio::test]
    async fn test_create_and_update_repository_with_empty_refs() {
        let conn = DbConn::new_in_memory().await.unwrap();

        // Insert new repository with empty refs
        conn.create_repository("test".into(), "testurl".into(), vec![])
            .await
            .unwrap();

        // Test that the refs defaults to None
        let repository = &conn
            .list_repositories_with_filter(None, None, false)
            .await
            .unwrap()[0];
        assert_eq!(repository.refs, None);

        // Update the repository with empty refs
        let id = repository.id;
        conn.update_repository(id, "test2".into(), "testurl2".into(), vec![])
            .await
            .unwrap();

        // Check the refs is still None
        let repository = &conn
            .list_repositories_with_filter(None, None, false)
            .await
            .unwrap()[0];
        assert_eq!(repository.refs, None);
    }
}
