

use anyhow::Result;
use sqlx::{
    query, query_as,
    sqlite::{SqliteConnectOptions},
    Pool, Sqlite, SqlitePool,
};
use tabby_common::Tag;

pub struct RepositoryCache {
    pool: Pool<Sqlite>,
}

struct RepositoryMetaDAO {
    git_url: String,
    filepath: String,
    language: String,
    max_line_length: i64,
    avg_line_length: f64,
    alphanum_fraction: f64,
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

    pub async fn get_repository_meta(
        &self,
        git_url: String,
        filepath: String,
    ) -> Result<RepositoryMetaDAO> {
        // TODO(boxbeam): Conversion from RepositoryMetaDAO to RepositoryMeta / SourceFile to never expose RepositoryMetaDAO
        let meta = query_as!(
            RepositoryMetaDAO,
            "SELECT git_url, filepath, language, max_line_length, avg_line_length, alphanum_fraction, tags FROM repository_meta WHERE git_url = ? AND filepath = ?",
            git_url, filepath
        ).fetch_one(&self.pool).await?;
        Ok(meta)
    }
}
