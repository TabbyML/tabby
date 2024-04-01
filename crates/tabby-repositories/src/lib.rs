use anyhow::Result;
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{query, SqlitePool};
use sqlx::{sqlite::SqliteConnectOptions, Pool, Sqlite};
use std::str::FromStr;
use tabby_common::Tag;

pub struct RepositoryCache {
    pool: Pool<Sqlite>,
}

struct RepositoryMetaDAO {
    git_url: String,
    filepath: String,
    language: String,
    max_line_length: usize,
    avg_line_length: f32,
    alphanum_fraction: f32,
    tags: String,
}

impl RepositoryCache {
    pub async fn new() -> Result<Self> {
        let init_query = include_str!("../schema.sql");
        let options = SqliteConnectOptions::new()
            .filename(tabby_common::path::repository_meta_db())
            .create_if_missing(true);
        let pool = SqlitePool::connect_with(options).await?;
        sqlx::query(init_query).execute(&pool).await?;
        Ok(RepositoryCache { pool })
    }

    pub async fn clear(&self) -> Result<()> {
        query!("DELETE FROM repository_meta")
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn add_repository_meta(
        &self,
        git_url: String,
        filepath: String,
        language: String,
        max_line_length: i64,
        avg_line_length: f32,
        alphanum_fraction: f32,
        tags: Vec<Tag>,
    ) -> Result<()> {
        let tags = serde_json::to_string(&tags)?;
        query!("INSERT INTO repository_meta (git_url, filepath, language, max_line_length, avg_line_length, alphanum_fraction, tags)
                VALUES ($1, $2, $3, $4, $5, $6, $7)",
                git_url, filepath, language, max_line_length, avg_line_length, alphanum_fraction, tags
        ).execute(&self.pool).await?;
        Ok(())
    }
}
