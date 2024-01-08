use anyhow::Result;

use crate::DbConn;

pub struct RepositoryDAO {
    pub name: String,
    pub git_url: String,
}

impl RepositoryDAO {
    fn new(name: String, git_url: String) -> Self {
        Self { name, git_url }
    }
}

impl DbConn {
    pub async fn list_repositories(&self) -> Result<Vec<RepositoryDAO>> {
        self.conn
            .call(|c| {
                let thing: Result<Vec<_>> = c
                    .prepare("SELECT name, git_url FROM repositories")?
                    .query_map([], |row| Ok(RepositoryDAO::new(row.get(1)?, row.get(2)?)))?
                    .map(|r| r.map_err(Into::into))
                    .collect();
                Ok(thing)
            })
            .await?
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
