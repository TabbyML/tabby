use anyhow::Result;
use chrono::{Duration, Utc};
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

#[derive(FromRow)]
pub struct JobStatsDAO {
    pub success: i32,
    pub failed: i32,
    pub pending: i32,
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
        jobs: Option<Vec<String>>,
        limit: Option<usize>,
        skip_id: Option<i32>,
        backwards: bool,
    ) -> Result<Vec<JobRunDAO>> {
        let mut conditions = vec![];

        if let Some(ids) = ids {
            let ids: Vec<String> = ids.iter().map(i32::to_string).collect();
            let ids = ids.join(", ");
            conditions.push(format!("id in ({ids})"));
        }

        if let Some(jobs) = jobs {
            let jobs: Vec<String> = jobs.iter().map(|s| format!("{s:?}")).collect();
            let jobs = jobs.join(", ");
            conditions.push(format!("job in ({jobs})"));
        }

        let condition = (!conditions.is_empty()).then_some(conditions.join(" AND "));
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

    pub async fn compute_job_stats(&self, jobs: Option<Vec<String>>) -> Result<JobStatsDAO> {
        let condition = match jobs {
            Some(jobs) => {
                let jobs: Vec<_> = jobs.into_iter().map(|s| format!("{s:?}")).collect();
                let jobs = jobs.join(", ");
                format!("AND job IN ({jobs})")
            }
            None => "".into(),
        };

        let cutoff = Utc::now() - Duration::days(7);

        let stats = sqlx::query_as(&format!(
            r#"SELECT
                SUM(exit_code == 0) AS success,
                SUM(exit_code != 0 AND exit_code IS NOT NULL) AS failed,
                SUM(exit_code IS NULL) AS pending FROM job_runs
                WHERE start_ts > ? {condition};"#
        ))
        .bind(cutoff)
        .fetch_one(&self.pool)
        .await?;
        Ok(stats)
    }

    pub async fn cleanup_stale_job_runs(&self) -> Result<()> {
        query!("DELETE FROM job_runs WHERE exit_code IS NULL;")
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}
