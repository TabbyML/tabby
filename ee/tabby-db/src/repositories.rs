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

    pub async fn create_repository(&self, name: String, git_url: String) -> Result<()> {
        Ok(self
            .conn
            .call(|c| {
                c.execute(
                    "INSERT INTO repositories (name, git_url) VALUES (?, ?)",
                    [name, git_url],
                )?;
                Ok(())
            })
            .await?)
    }

    pub async fn update_repository(&self, id: i32, name: String, git_url: String) -> Result<()> {
        let updated = self
            .conn
            .call(move |c| {
                let update_count = c.execute(
                    "UPDATE repositories SET git_url=?, name=? WHERE id=?",
                    (git_url, name, id),
                )?;
                Ok(update_count == 1)
            })
            .await?;
        if updated {
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
