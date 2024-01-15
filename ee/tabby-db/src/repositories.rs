use anyhow::{anyhow, Result};
use rusqlite::Row;
use sqlx::{
    query,
    sqlite::{SqliteConnectOptions, SqliteQueryResult},
    FromRow, Pool, Sqlite, SqlitePool,
};

use crate::DbConn;

#[derive(FromRow)]
pub struct RepositoryDAO {
    pub id: i32,
    pub name: String,
    pub git_url: String,
}

pub struct SQLXRepositoryService {
    conn: Pool<Sqlite>,
}

fn expect_single_change(result: SqliteQueryResult, error: &'static str) -> Result<()> {
    if result.rows_affected() == 1 {
        Ok(())
    } else {
        Err(anyhow!(error))
    }
}

impl SQLXRepositoryService {
    pub async fn new() -> Result<Self, sqlx::Error> {
        let options = SqliteConnectOptions::new().filename(crate::path::db_file());
        let conn = SqlitePool::connect_with(options).await?;
        Ok(Self { conn: conn.into() })
    }

    pub async fn create_repository(
        &self,
        name: String,
        git_url: String,
    ) -> Result<i32, sqlx::Error> {
        Ok(query!(
            "INSERT INTO repositories (name, git_url) VALUES (?, ?)",
            name,
            git_url
        )
        .execute(&self.conn)
        .await?
        .last_insert_rowid() as i32)
    }

    pub async fn delete_repository(&self, id: i32) -> Result<()> {
        let result = query!("DELETE FROM repositories WHERE id = ?", id)
            .execute(&self.conn)
            .await?;
        expect_single_change(result, "failed to delete repository: ID not found")
    }

    pub async fn list_repositories_with_filter(
        &self,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<RepositoryDAO>> {
        let query = DbConn::make_pagination_query(
            "repositories",
            &["id", "name", "git_url"],
            limit,
            skip_id,
            backwards,
        );
        let rows = sqlx::query(&query)
            .try_map(|r| FromRow::from_row(&r))
            .fetch_all(&self.conn)
            .await?;
        Ok(rows)
    }

    pub async fn update_repository(&self, id: i32, name: String, git_url: String) -> Result<()> {
        let result = query!(
            "UPDATE repositories SET name = ?, git_url = ? WHERE id = ?",
            name,
            git_url,
            id
        )
        .execute(&self.conn)
        .await?;
        expect_single_change(result, "failed to update repository: ID not found")
    }
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
