use anyhow::Result;
use rusqlite::{OptionalExtension, Row};

use crate::DbConn;

pub struct RepositoryDAO {
    pub name: String,
    pub git_url: String,
}

impl RepositoryDAO {
    fn new(name: String, git_url: String) -> Self {
        Self { name, git_url }
    }

    fn from_row(row: &Row) -> Result<Self, rusqlite::Error> {
        Ok(Self::new(row.get(0)?, row.get(1)?))
    }
}

impl DbConn {
    pub async fn list_repositories(&self) -> Result<Vec<RepositoryDAO>> {
        self.conn
            .call(|c| {
                let thing: Result<Vec<_>> = c
                    .prepare("SELECT name, git_url FROM repositories")?
                    .query_map([], RepositoryDAO::from_row)?
                    .map(|r| r.map_err(Into::into))
                    .collect();
                Ok(thing)
            })
            .await?
    }

    pub async fn get_repository(&self, name: String) -> Result<Option<RepositoryDAO>> {
        Ok(self
            .conn
            .call(|c| {
                Ok(c.query_row(
                    "SELECT name, git_url FROM repositories WHERE name=?",
                    [name],
                    RepositoryDAO::from_row,
                ))
            })
            .await?
            .optional()?)
    }

    pub async fn delete_repository(&self, name: String) -> Result<bool> {
        Ok(self
            .conn
            .call(|c| {
                let deleted = c
                    .execute("DELETE FROM repositories WHERE name=?", [name])
                    .is_ok();
                Ok(deleted)
            })
            .await?)
    }

    pub async fn add_repository(&self, name: String, git_url: String) -> Result<()> {
        Ok(self
            .conn
            .call(|c| {
                c.execute("INSERT INTO repositories VALUES (?, ?)", [name, git_url])?;
                Ok(())
            })
            .await?)
    }

    pub async fn update_repository_url(&self, name: String, git_url: String) -> Result<bool> {
        Ok(self
            .conn
            .call(|c| {
                let updated = c.execute(
                    "UPDATE repositories SET git_url=? WHERE name=?",
                    [git_url, name],
                )?;
                Ok(updated == 1)
            })
            .await?)
    }
}

#[cfg(test)]
mod tests {
    use crate::DbConn;

    #[tokio::test]
    async fn test_update_repository() {
        let conn = DbConn::new_in_memory().await.unwrap();

        conn.add_repository("test".into(), "testurl".into())
            .await
            .unwrap();

        let repository = conn.get_repository("test".into()).await.unwrap().unwrap();
        assert_eq!(repository.git_url, "testurl");

        conn.update_repository_url("test".into(), "testurl2".into())
            .await
            .unwrap();

        let repository = conn.get_repository("test".into()).await.unwrap().unwrap();
        assert_eq!(repository.git_url, "testurl2");
    }
}
