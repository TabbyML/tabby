use anyhow::Result;
use chrono::{DateTime, Utc};

use super::DbConn;

#[derive(Default, Clone)]
pub struct JobRunDAO {
    pub id: i32,
    pub job_name: String,
    pub start_time: DateTime<Utc>,
    pub finish_time: Option<DateTime<Utc>>,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

impl JobRunDAO {
    fn from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            id: row.get(0)?,
            job_name: row.get(1)?,
            start_time: row.get(2)?,
            finish_time: row.get(3)?,
            exit_code: row.get(4)?,
            stdout: row.get(5)?,
            stderr: row.get(6)?,
        })
    }
}

/// db read/write operations for `job_runs` table
impl DbConn {
    pub async fn create_job_run(&self, run: JobRunDAO) -> Result<i32> {
        let rowid = self
            .conn
            .call(move |c| {
                let mut stmt = c.prepare(
                    r#"INSERT INTO job_runs (job, start_ts, end_ts, exit_code, stdout, stderr) VALUES (?, ?, ?, ?, ?, ?)"#,
                )?;
                let rowid = stmt.insert((
                    run.job_name,
                    run.start_time,
                    run.finish_time,
                    run.exit_code,
                    run.stdout,
                    run.stderr,
                ))?;
                Ok(rowid)
            })
            .await?;

        Ok(rowid as i32)
    }

    pub async fn update_job_stdout(&self, job_id: i32, stdout: String) -> Result<()> {
        self.conn
            .call(move |c| {
                let mut stmt = c.prepare(
                    r#"UPDATE job_runs SET stdout = stdout || ?, updated_at = datetime('now') WHERE id = ?"#,
                )?;
                stmt.execute((stdout, job_id))?;
                Ok(())
            })
            .await?;
        Ok(())
    }

    pub async fn update_job_stderr(&self, job_id: i32, stderr: String) -> Result<()> {
        self.conn
            .call(move |c| {
                let mut stmt = c.prepare(
                    r#"UPDATE job_runs SET stderr = stderr || ?, updated_at = datetime('now') WHERE id = ?"#,
                )?;
                stmt.execute((stderr, job_id))?;
                Ok(())
            })
            .await?;
        Ok(())
    }

    pub async fn update_job_status(&self, run: JobRunDAO) -> Result<()> {
        self.conn
            .call(move |c| {
                let mut stmt = c.prepare(
                    r#"UPDATE job_runs SET end_ts = ?, exit_code = ?, updated_at = datetime('now') WHERE id = ?"#,
                )?;
                stmt.execute((
                    run.finish_time,
                    run.exit_code,
                    run.id,
                ))?;
                Ok(())
            })
            .await?;
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
                "start_ts",
                "end_ts",
                "exit_code",
                "stdout",
                "stderr",
            ],
            limit,
            skip_id,
            backwards,
        );

        let runs = self
            .conn
            .call(move |c| {
                let mut stmt = c.prepare(&query)?;
                let run_iter = stmt.query_map([], JobRunDAO::from_row)?;
                Ok(run_iter.filter_map(|x| x.ok()).collect::<Vec<_>>())
            })
            .await?;

        Ok(runs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_job_run() {
        let db = DbConn::new_in_memory().await.unwrap();
        let run = JobRunDAO {
            id: 0,
            job_name: "test".to_string(),
            start_time: chrono::Utc::now(),
            finish_time: None,
            exit_code: None,
            stdout: "stdout".to_string(),
            stderr: "stderr".to_string(),
        };
        let id = db.create_job_run(run).await.unwrap();
        assert_eq!(id, 1);

        let run = JobRunDAO {
            id: 0,
            job_name: "test".to_string(),
            start_time: chrono::Utc::now(),
            finish_time: Some(chrono::Utc::now()),
            exit_code: None,
            stdout: "stdout".to_string(),
            stderr: "stderr".to_string(),
        };
        let id = db.create_job_run(run).await.unwrap();
        assert_eq!(id, 2);

        let run = JobRunDAO {
            id: 0,
            job_name: "test".to_string(),
            start_time: chrono::Utc::now(),
            finish_time: Some(chrono::Utc::now()),
            exit_code: Some(0),
            stdout: "stdout".to_string(),
            stderr: "stderr".to_string(),
        };
        let id = db.create_job_run(run).await.unwrap();
        assert_eq!(id, 3);
    }
}
