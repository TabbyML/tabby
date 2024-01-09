use anyhow::Result;
use rusqlite::{OptionalExtension, Params, Row};
use serde::{Deserialize, Serialize};

use crate::DbConn;

#[derive(Serialize, Deserialize)]
pub struct RepositoryDAO {
    pub id: i32,
    pub name: String,
    pub git_url: String,
}

impl DbConn {
    pub async fn execute(
        &self,
        query: &'static str,
        params: impl Params + Send + 'static,
    ) -> Result<usize> {
        Ok(self
            .conn
            .call(move |c| Ok(c.execute(query, params)))
            .await??)
    }

    pub async fn select_one<T: for<'a> Deserialize<'a> + Send + 'static>(
        &self,
        query: &'static str,
        params: impl Params + Send + 'static,
    ) -> Result<T> {
        let row = self.conn.call(move |c| {
            let res = c.query_row(query, params, |r| Ok(serde_rusqlite::from_row(r).unwrap()));
            Ok(res?)
        });
        Ok(row.await?)
    }

    pub async fn select_one_optional<T: for<'a> Deserialize<'a> + Send + 'static>(
        &self,
        query: &'static str,
        params: impl Params + Send + 'static,
    ) -> Result<Option<T>> {
        let row = self.conn.call(move |c| {
            let res = c
                .query_row(query, params, |r| {
                    Ok(serde_rusqlite::from_row::<T>(r).unwrap())
                })
                .optional();
            Ok(res?)
        });
        Ok(row.await?)
    }

    pub async fn select_list<T: for<'a> Deserialize<'a> + Send + 'static>(
        &self,
        query: &'static str,
        params: impl Params + Send + 'static,
    ) -> Result<Vec<T>> {
        let row = self.conn.call(move |c| {
            let mut stmt = c.prepare(query)?;
            let res = stmt.query_map(params, |r| Ok(serde_rusqlite::from_row(r).unwrap()));
            let mut elems = vec![];
            for elem in res? {
                elems.push(elem?);
            }
            Ok(elems)
        });
        Ok(row.await?)
    }
}

impl DbConn {
    pub async fn list_repositories(&self) -> Result<Vec<RepositoryDAO>> {
        self.select_list("SELECT * FROM repositories", []).await
    }

    pub async fn get_repository(&self, id: i32) -> Result<Option<RepositoryDAO>> {
        self.select_one_optional("SELECT * FROM repositories WHERE id=?", [id])
            .await
    }

    pub async fn get_repository_by_name(&self, name: String) -> Result<Option<RepositoryDAO>> {
        self.select_one_optional("SELECT * FROM repositories WHERE name=?", [name])
            .await
    }

    pub async fn delete_repository(&self, id: i32) -> Result<bool> {
        self.execute("DELETE FROM repositories WHERE id=?", [id])
            .await
            .map(|n| n == 1)
    }

    pub async fn add_repository(&self, name: String, git_url: String) -> Result<()> {
        self.execute("INSERT INTO repositories VALUES (?, ?)", [name, git_url])
            .await
            .map(|_| ())
    }

    pub async fn update_repository_url(&self, id: i32, git_url: String) -> Result<bool> {
        self.execute(
            "UPDATE repositories SET git_url=? WHERE id=?",
            (git_url, id),
        )
        .await
        .map(|n| n == 1)
    }
}

#[cfg(test)]
mod tests {
    use crate::DbConn;

    #[tokio::test]
    async fn test_update_repository() {
        let conn = DbConn::new_in_memory().await.unwrap();

        // Insert new repository
        conn.add_repository("test".into(), "testurl".into())
            .await
            .unwrap();

        // Test that the url can be retrieved
        let repository = conn
            .get_repository_by_name("test".into())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(repository.git_url, "testurl");

        // Update the repository
        let id = repository.id;
        conn.update_repository_url(id, "testurl2".into())
            .await
            .unwrap();

        // Check the url was updated
        let repository = conn.get_repository(id).await.unwrap().unwrap();
        assert_eq!(repository.git_url, "testurl2");
    }
}
