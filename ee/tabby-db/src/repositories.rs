use anyhow::{anyhow, Result};
use rusqlite::Row;

use crate::DbConn;

pub struct RepositoryDAO {
    pub id: i32,
    pub name: String,
    pub git_url: String,
}

impl RepositoryDAO {
    fn new(id: i32, name: String, git_url: String) -> Self {
        Self { id, name, git_url }
    }

    fn from_row(row: &Row) -> Result<Self, rusqlite::Error> {
        Ok(Self::new(row.get(0)?, row.get(1)?, row.get(2)?))
    }
}

impl DbConn {
    pub async fn list_repositories_with_filter(
        &self,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<RepositoryDAO>> {
        let query = Self::make_pagination_query(
            "repositories",
            &["id", "name", "git_url"],
            limit,
            skip_id,
            backwards,
        );

        self.conn
            .call(move |c| {
                let thing: Result<Vec<_>> = c
                    .prepare(&query)?
                    .query_map([], RepositoryDAO::from_row)?
                    .map(|r| r.map_err(Into::into))
                    .collect();
                Ok(thing)
            })
            .await?
    }

    pub async fn delete_repository(&self, id: i32) -> Result<bool> {
        Ok(self
            .conn
            .call(move |c| {
                let deleted = c
                    .execute("DELETE FROM repositories WHERE id=?", [id])
                    .is_ok();
                Ok(deleted)
            })
            .await?)
    }

    pub async fn create_repository(&self, name: String, git_url: String) -> Result<i32> {
        Ok(self
            .conn
            .call(|c| {
                let id = c
                    .prepare("INSERT INTO repositories (name, git_url) VALUES (?, ?)")?
                    .insert([name, git_url])?;
                Ok(id as i32)
            })
            .await?)
    }
}

#[cfg(test)]
mod tests {
    use crate::DbConn;

    #[tokio::test]
    async fn test_create_repository() {
        let conn = DbConn::new_in_memory().await.unwrap();

        // Insert new repository
        let id = conn
            .create_repository("test".into(), "testurl".into())
            .await
            .unwrap();

        // Test that the url can be retrieved
        let repository = &conn
            .list_repositories_with_filter(None, None, false)
            .await
            .unwrap()[0];
        assert_eq!(repository.git_url, "testurl");

        // Delete the repository and test it is deleted successfully
        assert!(conn.delete_repository(id).await.unwrap());
        assert!(conn
            .list_repositories_with_filter(None, None, false)
            .await
            .unwrap()
            .is_empty());
    }
}
