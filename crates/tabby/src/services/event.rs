use std::{path::PathBuf, time::Duration};

use chrono::Utc;
use lazy_static::lazy_static;
use tabby_common::{api::event::RawEventLogger, path};
use tokio::{
    io::AsyncWriteExt,
    sync::mpsc::{unbounded_channel, UnboundedSender},
    time::{self},
};

lazy_static! {
    static ref WRITER: UnboundedSender<String> = {
        let (tx, mut rx) = unbounded_channel::<String>();

        tokio::spawn(async move {
            let mut writer = EventWriter::new(path::events_dir()).await;
            let mut interval = time::interval(Duration::from_secs(5));

            loop {
                tokio::select! {
                    content = rx.recv() => {
                        if let Some(content) = content {
                            writer.write_line(content).await;
                        } else {
                            break;
                        }
                    }
                    _ = interval.tick() => {
                        writer.flush().await;
                    }
                }
            }
        });

        tx
    };
}

struct EventWriter {
    events_dir: PathBuf,
    filename: Option<String>,
    writer: Option<tokio::io::BufWriter<tokio::fs::File>>,
}

impl EventWriter {
    async fn new(events_dir: PathBuf) -> Self {
        tokio::fs::create_dir_all(events_dir.as_path()).await.ok();

        Self {
            events_dir,
            filename: None,
            writer: None,
        }
    }

    #[cfg(test)]
    fn event_file_path(&self) -> Option<PathBuf> {
        self.filename
            .as_ref()
            .map(|fname| self.events_dir.join(fname))
    }

    async fn write_line(&mut self, content: String) {
        let now = Utc::now();
        let fname = now.format("%Y-%m-%d.json");

        if self.filename != Some(fname.to_string()) {
            if let Some(mut w) = self.writer.take() {
                w.flush().await.unwrap();
            }

            let file = tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .write(true)
                .open(self.events_dir.join(fname.to_string()))
                .await
                .ok()
                .unwrap();
            self.writer = Some(tokio::io::BufWriter::new(file));
            self.filename = Some(fname.to_string());
        }

        let writer = self.writer.as_mut().unwrap();
        writer
            .write_all(format!("{}\n", content).as_bytes())
            .await
            .unwrap();
    }

    async fn flush(&mut self) {
        let writer = self.writer.as_mut().unwrap();
        writer.flush().await.unwrap()
    }
}

struct EventService;

impl RawEventLogger for EventService {
    fn log(&self, content: String) {
        WRITER.send(content).unwrap();
    }
}

pub fn create_logger() -> impl RawEventLogger {
    EventService
}

#[cfg(test)]
mod tests {
    use super::*;

    fn events_dir() -> PathBuf {
        std::env::temp_dir().join(".tabby").join("events")
    }

    async fn test_event_writer_swap_file() {
        tokio::fs::create_dir_all(events_dir()).await.ok();

        let old_fname = "2021-01-01.json".to_string();
        let old_file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .write(true)
            .open(events_dir().join(old_fname.clone()))
            .await
            .ok()
            .unwrap();
        let mut old_wr = tokio::io::BufWriter::new(old_file);
        old_wr
            .write_all(format!("{}\n", "old data in old file").as_bytes())
            .await
            .unwrap();
        old_wr.flush().await.unwrap();

        let mut event_wr = EventWriter {
            events_dir: events_dir(),
            filename: Some(old_fname.clone()),
            writer: Some(old_wr),
        };
        event_wr.write_line("test data".to_string()).await;
        event_wr.flush().await;

        // we should be able to read new created file successfully
        let content = tokio::fs::read_to_string(event_wr.event_file_path().unwrap())
            .await
            .unwrap();
        assert_eq!(content.as_str(), "test data\n");
        // old file should have no more writes
        let content = tokio::fs::read_to_string(events_dir().join(old_fname))
            .await
            .unwrap();
        assert_eq!(content.as_str(), "old data in old file\n");
    }

    #[tokio::test]
    async fn test_event_writer() {
        // in case previous test failed
        tokio::fs::remove_dir_all(events_dir()).await.ok();

        test_event_writer_swap_file().await;
        tokio::fs::remove_dir_all(events_dir()).await.unwrap();
    }
}
