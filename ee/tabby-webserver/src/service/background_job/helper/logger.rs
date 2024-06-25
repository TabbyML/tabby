use tabby_db::DbConn;
use tracing::warn;

#[derive(Clone)]
pub struct JobLoggerGuard;

impl JobLoggerGuard {
    pub fn new(db: DbConn, id: i64) -> Self {
        let logger = DbLogger::new(db, id);
        let _ = log::set_boxed_logger(Box::new(logger));
        Self
    }
}

impl Drop for JobLoggerGuard {
    fn drop(&mut self) {
        let _ = log::set_logger(&NoneLogger);
    }
}

#[derive(Clone)]
struct DbLogger {
    id: i64,
    db: DbConn,
}

impl DbLogger {
    fn new(db: DbConn, id: i64) -> Self {
        Self { id, db }
    }
}

impl log::Log for DbLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record) {
        let id = self.id;
        let logger = self.clone();
        let stdout = format!("{}: {}\n", record.level(), record.args());
        tokio::spawn(async move {
            match logger.db.update_job_stdout(id, stdout).await {
                Ok(_) => (),
                Err(_) => {
                    warn!("Failed to write stdout to job `{}`", id);
                }
            }
        });
    }

    fn flush(&self) {}
}

struct NoneLogger;

impl log::Log for NoneLogger {
    fn enabled(&self, _: &log::Metadata) -> bool { false }
    fn log(&self, _: &log::Record) {}
    fn flush(&self) {}
}
