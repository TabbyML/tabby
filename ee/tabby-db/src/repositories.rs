use anyhow::{anyhow, Result};
use sqlx::{prelude::FromRow, query};

use crate::{make_pagination_query, DbConn, SQLXResultExt};

#[derive(FromRow)]
pub struct RepositoryDAO {
    pub id: i32,
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
        let query = make_pagination_query(
            "repositories",
            &["id", "name", "git_url"],
            limit,
            skip_id,
            backwards,
        );

        let repos = sqlx::query_as(&query).fetch_all(&self.pool).await?;
        Ok(repos)
    }

    pub async fn delete_repository(&self, id: i32) -> Result<bool> {
        let res = query!("DELETE FROM repositories WHERE id = ?", id)
            .execute(&self.pool)
            .await?;
        Ok(res.rows_affected() == 1)
    }

    pub async fn create_repository(&self, name: String, git_url: String) -> Result<i32> {
        let res = query!(
            "INSERT INTO repositories (name, git_url) VALUES (?, ?)",
            name,
            git_url
        )
        .execute(&self.pool)
        .await;

        res.unique_error("A repository with the same name or URL already exists")
            .map(|output| output.last_insert_rowid() as i32)
    }

    pub async fn update_repository(&self, id: i32, name: String, git_url: String) -> Result<()> {
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
