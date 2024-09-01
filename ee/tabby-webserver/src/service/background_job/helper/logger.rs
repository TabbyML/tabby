use tabby_db::DbConn;
use tracing::warn;

pub struct JobLogger {
    handle: tokio::task::JoinHandle<()>,
}

impl JobLogger {
    pub fn new(db: DbConn, id: i64) -> Self {
        let mut logger = logkit::Logger::new(None);
        logger.mount(logkit::LevelPlugin);
        logger.mount(logkit::TimePlugin::from_micros());
        let (target, handle) = DbTarget::new(db, id);
        logger.route(target);
        logkit::set_default_logger(logger);
        Self { handle }
    }

    pub async fn finalize(self) {
        logkit::set_default_logger(logkit::Logger::new(None));
        self.handle.await.unwrap_or_else(|err| {
            warn!("Failed to join logging thread: {}", err);
        });
    }
}

struct DbTarget {
    tx: tokio::sync::mpsc::Sender<Record>,
}

impl DbTarget {
    fn new(db: DbConn, id: i64) -> (Self, tokio::task::JoinHandle<()>) {
        let (tx, rx) = tokio::sync::mpsc::channel::<Record>(1024);
        let handle = Self::create_logging_thread(db, id, rx);
        (Self { tx }, handle)
    }
}

impl DbTarget {
    fn create_logging_thread(
        db: DbConn,
        id: i64,
        mut rx: tokio::sync::mpsc::Receiver<Record>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            while let Some(record) = rx.recv().await {
                let stdout = format!(
                    "{} [{}]: {}\n",
                    record.time,
                    record.level.to_uppercase(),
                    record.msg
                );

                match db.update_job_stdout(id, stdout).await {
                    Ok(_) => (),
                    Err(_) => {
                        warn!("Failed to write stdout to job `{}`", id);
                    }
                }

                if let Some(exit_code) = record.exit_code {
                    match db.update_job_status(id, exit_code).await {
                        Ok(_) => (),
                        Err(_) => {
                            warn!("Failed to write exit code to job `{}`", id);
                        }
                    }
                }
            }
        })
    }
}

#[derive(serde::Deserialize)]
struct Record {
    level: String,
    time: String,
    msg: String,
    exit_code: Option<i32>,
}

impl logkit::Target for DbTarget {
    fn write(&self, buf: &[u8]) {
        let Ok(record) = serde_json::from_slice::<Record>(buf) else {
            warn!("Failed to parse log record");
            return;
        };

        self.tx.try_send(record).unwrap_or_else(|err| {
            warn!("Failed to send log record: {}", err);
        });
    }
}
