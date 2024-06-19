use tabby_db::DbConn;
use tracing::warn;

#[derive(Clone)]
pub struct JobLogger {
    id: i64,
    db: DbConn,
}

impl JobLogger {
    pub async fn new(name: &str, params: Option<&str>, db: DbConn) -> Self {
        let id = db
            .create_job_run(name.to_owned(), params.map(str::to_string))
            .await
            .expect("failed to create job");
        Self { id, db }
    }

    pub async fn r#internal_println(&self, stdout: String) {
        let stdout = stdout + "\n";
        match self.db.update_job_stdout(self.id, stdout).await {
            Ok(_) => (),
            Err(_) => {
                warn!("Failed to write stdout to job `{}`", self.id);
            }
        }
    }

    pub async fn complete(&mut self, exit_code: i32) {
        match self.db.update_job_status(self.id, exit_code).await {
            Ok(_) => (),
            Err(_) => {
                warn!("Failed to complete job `{}`", self.id);
            }
        }
    }
}
