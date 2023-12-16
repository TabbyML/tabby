use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::DbConn;

#[derive(Serialize, Deserialize)]
pub struct JobRun {
    pub id: i32,
    pub job_name: String,
    pub start_time: DateTime<Utc>,
    pub finish_time: Option<DateTime<Utc>>,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

/// db read/write operations for `job_runs` table
impl DbConn {
    pub async fn create_job_run(&self, run: JobRun) -> Result<i32> {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_job_run() {
        let db = DbConn::new_in_memory().await.unwrap();
        let run = JobRun {
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

        let run = JobRun {
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

        let run = JobRun {
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
