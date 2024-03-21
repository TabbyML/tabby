use anyhow::Result;
use sqlx::{query, FromRow};
use tabby_db_macros::query_paged_as;

use super::DbConn;
use crate::{DateTimeUtc, DbOption};

#[derive(Default, Clone, FromRow)]
pub struct JobRunDAO {
    pub id: i64,
    #[sqlx(rename = "job")]
    pub name: String,
    pub exit_code: Option<i64>,
    pub stdout: String,
    pub stderr: String,
    pub created_at: DateTimeUtc,
    pub updated_at: DateTimeUtc,

    #[sqlx(rename = "end_ts")]
    pub finished_at: DbOption<DateTimeUtc>,
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
        ids: Option<Vec<i32>>,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<JobRunDAO>> {
        let condition = if let Some(ids) = ids {
            let ids: Vec<String> = ids.iter().map(i32::to_string).collect();
            let ids = ids.join(", ");
            Some(format!("id in ({ids})"))
        } else {
            None
        };
        let job_runs: Vec<JobRunDAO> = query_paged_as!(
            JobRunDAO,
            "job_runs",
            [
                "id",
                "job" as "name",
                "exit_code",
                "stdout",
                "stderr",
                "created_at"!,
                "updated_at"!,
                "end_ts" as "finished_at"
            ],
            limit,
            skip_id,
            backwards,
            condition
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(job_runs)
    }

    pub async fn cleanup_stale_job_runs(&self) -> Result<()> {
        query!("DELETE FROM job_runs WHERE exit_code IS NULL;")
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}
