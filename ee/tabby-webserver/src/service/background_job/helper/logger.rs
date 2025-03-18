use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
};

use anyhow::Result;
use chrono::{DateTime, Local};
use tabby_db::DbConn;
use tracing::warn;

use crate::path::background_jobs_dir;

pub struct JobLogger {
    handle: tokio::task::JoinHandle<()>,
}

impl JobLogger {
    pub fn new(db: DbConn, id: i64) -> Result<Self> {
        let mut logger = logkit::Logger::new(None);
        logger.mount(logkit::LevelPlugin);
        logger.mount(logkit::TimePlugin::from_micros());

        let (target, handle) = DbTarget::new(db, id)?;
        logger.route(target);

        logkit::set_default_logger(logger);
        Ok(Self { handle })
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
    fn new(db: DbConn, id: i64) -> Result<(Self, tokio::task::JoinHandle<()>)> {
        let job_dir = background_jobs_dir().join(format!("{}", id));
        std::fs::create_dir_all(&job_dir)?;

        let (tx, rx) = tokio::sync::mpsc::channel::<Record>(1024);
        let handle = Self::create_logging_thread(db, id, &job_dir.to_string_lossy(), "stdout", rx);
        Ok((Self { tx }, handle))
    }

    fn create_logging_thread(
        db: DbConn,
        id: i64,
        dir: &str,
        name: &str,
        mut rx: tokio::sync::mpsc::Receiver<Record>,
    ) -> tokio::task::JoinHandle<()> {
        let dir = dir.to_owned();
        let name = name.to_owned();
        tokio::spawn(async move {
            let mut last_rotation = Local::now();
            let mut writer = match Self::create_log_file(dir.clone(), name.clone()) {
                Ok(writer) => writer,
                Err(err) => {
                    warn!("Failed to create log file: {}", err);
                    return;
                }
            };

            while let Some(record) = rx.recv().await {
                let stdout = format!(
                    "{} [{}]: {}\n",
                    record.time,
                    record.level.to_uppercase(),
                    record.msg
                );

                if should_rotate(Local::now(), last_rotation) {
                    last_rotation = Local::now();
                    writer = match Self::create_log_file(dir.clone(), name.clone()) {
                        Ok(writer) => writer,
                        Err(err) => {
                            warn!("Failed to create log file: {}", err);
                            continue;
                        }
                    };
                }

                if let Err(err) = writer.write_all(stdout.as_bytes()) {
                    warn!("Failed to write log record to file: {}", err);
                }
                if let Err(err) = writer.flush() {
                    warn!("Failed to flush log buffer: {}", err);
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

    fn create_log_file(dir: String, name: String) -> Result<BufWriter<File>> {
        let now = Local::now();
        let filename = format!("{}/{}_{}.log", dir, name, now.format("%Y-%m-%d_%H-%M-%S"));

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filename)?;

        Ok(BufWriter::new(file))
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

fn should_rotate(now: DateTime<Local>, last_rotation: DateTime<Local>) -> bool {
    last_rotation
        .date_naive()
        .signed_duration_since(now.date_naive())
        .num_days()
        .abs()
        >= 1
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    #[test]
    fn test_should_rotate() {
        let cases = [
            (
                Local.with_ymd_and_hms(2025, 1, 2, 0, 0, 0).unwrap(),
                Local.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap(),
                true,
            ),
            (
                Local.with_ymd_and_hms(2025, 1, 2, 0, 0, 0).unwrap(),
                Local.with_ymd_and_hms(2025, 1, 2, 0, 0, 0).unwrap(),
                false,
            ),
            (
                Local.with_ymd_and_hms(2025, 1, 2, 0, 0, 0).unwrap(),
                Local.with_ymd_and_hms(2025, 1, 1, 23, 59, 59).unwrap(),
                true,
            ),
        ];

        for (now, last_rotation, expected) in cases {
            assert_eq!(should_rotate(now, last_rotation), expected);
        }
    }
}
