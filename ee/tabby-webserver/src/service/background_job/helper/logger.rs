use juniper::ID;
use tabby_db::DbConn;
use tracing::warn;

#[derive(Clone)]
pub struct JobLogger {
    id: i64,
    db: DbConn,
}

impl JobLogger {
    pub async fn new(db: DbConn, id: i64) -> Self {
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
