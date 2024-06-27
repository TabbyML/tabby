use anyhow::Result;
use chrono::Duration;
use sqlx::{query, FromRow};
use tabby_db_macros::query_paged_as;

use super::DbConn;
use crate::{DateTimeUtc, DbOption};

#[derive(Default, Clone, FromRow)]
pub struct JobRunDAO {
    pub id: i64,
    #[sqlx(rename = "job")]
    pub name: String,
    pub command: String,
    pub exit_code: Option<i64>,
    pub stdout: String,
    pub stderr: String,
    pub created_at: DateTimeUtc,
    pub updated_at: DateTimeUtc,

    /// The time when the job was started.
    pub started_at: DbOption<DateTimeUtc>,

    #[sqlx(rename = "end_ts")]
    pub finished_at: DbOption<DateTimeUtc>,
}

impl JobRunDAO {
    pub fn is_running(&self) -> bool {
        self.started_at.0.is_some() && self.finished_at.0.is_none()
    }

    pub fn is_pending(&self) -> bool {
        self.started_at.0.is_none()
    }

    pub fn is_finished(&self) -> bool {
        self.finished_at.0.is_some()
    }
}

#[derive(FromRow)]
pub struct JobStatsDAO {
    pub success: i32,
    pub failed: i32,
    pub pending: i32,
}

/// db read/write operations for `job_runs` table
impl DbConn {
    pub async fn create_job_run(&self, job: String, command: String) -> Result<i64> {
        let rowid = query!(
            r#"INSERT INTO job_runs (job, start_ts, stdout, stderr, command) VALUES (?, DATETIME('now'), '', '', ?)"#,
            job, command,
        ).execute(&self.pool).await?.last_insert_rowid();

        Ok(rowid)
    }

    pub async fn get_next_job_to_execute(&self) -> Option<JobRunDAO> {
        sqlx::query_as(
            r#"SELECT * FROM job_runs WHERE exit_code IS NULL AND started_at is NULL ORDER BY created_at ASC LIMIT 1"#,
        )
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten()
    }

    pub async fn update_job_stdout(&self, job_id: i64, stdout: String) -> Result<()> {
        query!(
            r#"UPDATE job_runs SET stdout = stdout || ?, updated_at = datetime('now') WHERE id = ?"#,
            stdout,
            job_id
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn update_job_stderr(&self, job_id: i64, stderr: String) -> Result<()> {
        query!(
            r#"UPDATE job_runs SET stderr = stderr || ?, updated_at = datetime('now') WHERE id = ?"#,
            stderr,
            job_id
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn update_job_status(&self, job_id: i64, exit_code: i32) -> Result<()> {
        query!(
            r#"UPDATE job_runs SET end_ts = datetime('now'), exit_code = ?, updated_at = datetime('now') WHERE id = ?"#,
            exit_code,
            job_id,
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn update_job_started(&self, job_id: i64) -> Result<()> {
        query!(
            r#"UPDATE job_runs SET started_at = datetime('now'), updated_at = datetime('now') WHERE id = ?"#,
            job_id,
        ).execute(&self.pool).await?;
        Ok(())
    }

    pub async fn get_latest_job_run(&self, command: String) -> Option<JobRunDAO> {
        sqlx::query_as(
            r#"SELECT * FROM job_runs WHERE command = ? ORDER BY created_at DESC LIMIT 1"#,
        )
        .bind(command)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten()
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
                "command"!,
                "created_at"!,
                "updated_at"!,
                "started_at",
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

        let cutoff = DateTimeUtc::now() - Duration::days(7);

        let stats = sqlx::query_as(&format!(
            r#"SELECT
                SUM(exit_code == 0) AS success,
                SUM(exit_code != 0 AND exit_code IS NOT NULL) AS failed,
                SUM(exit_code IS NULL) AS pending FROM job_runs
                WHERE created_at > ? {condition};"#
        ))
        .bind(cutoff)
        .fetch_one(&self.pool)
        .await?;
        Ok(stats)
    }

    pub async fn finalize_stale_job_runs(&self) -> Result<()> {
        query!("UPDATE job_runs SET exit_code = -1, started_at = datetime('now'), end_ts = datetime('now'), updated_at = datetime('now') WHERE exit_code IS NULL;")
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}
