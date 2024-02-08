use anyhow::Result;
use chrono::{DateTime, Utc};
use sqlx::{query, FromRow};

use super::DbConn;

#[derive(Default, Clone, FromRow)]
pub struct JobRunDAO {
    pub id: i32,
    pub job: String,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,

    #[sqlx(rename = "end_ts")]
    pub finished_at: Option<DateTime<Utc>>,
}

/// db read/write operations for `job_runs` table
impl DbConn {
    pub async fn create_job_run(&self, job: String) -> Result<i32> {
        let rowid = query!(
            r#"INSERT INTO job_runs (job, start_ts, stdout, stderr) VALUES (?, DATETIME('now'), '', '')"#,
            job,
        ).execute(&self.pool).await?.last_insert_rowid();

        Ok(rowid as i32)
    }

    pub async fn update_job_stdout(&self, job_id: i32, stdout: String) -> Result<()> {
        query!(
            r#"UPDATE job_runs SET stdout = stdout || ?, updated_at = datetime('now') WHERE id = ?"#,
            stdout,
            job_id
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn update_job_stderr(&self, job_id: i32, stderr: String) -> Result<()> {
        query!(
            r#"UPDATE job_runs SET stderr = stderr || ?, updated_at = datetime('now') WHERE id = ?"#,
            stderr,
            job_id
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn update_job_status(&self, job_id: i32, exit_code: i32) -> Result<()> {
        query!(
            r#"UPDATE job_runs SET end_ts = datetime('now'), exit_code = ?, updated_at = datetime('now') WHERE id = ?"#,
            exit_code,
            job_id,
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn list_job_runs_with_filter(
        &self,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<JobRunDAO>> {
        let query = Self::make_pagination_query(
            "job_runs",
            &[
                "id",
                "job",
                "exit_code",
                "stdout",
                "stderr",
                "created_at",
                "updated_at",
                "end_ts",
            ],
            limit,
            skip_id,
            backwards,
        );

        let runs = sqlx::query_as(&query).fetch_all(&self.pool).await?;
        Ok(runs)
    }
}
