use std::{
    fs,
    io::{BufWriter, Write},
    time::Duration,
};

use chrono::Utc;
use lazy_static::lazy_static;
use serde::Serialize;
use tabby_common::path;
use tokio::{
    sync::mpsc::{unbounded_channel, UnboundedSender},
    time::{self},
};

use crate::api::{Event, EventLogger};

lazy_static! {
    static ref WRITER: UnboundedSender<String> = {
        let (tx, mut rx) = unbounded_channel::<String>();

        tokio::spawn(async move {
            let events_dir = path::events_dir();
            std::fs::create_dir_all(events_dir.as_path()).ok();

            let now = Utc::now();
            let fname = now.format("%Y-%m-%d.json");
            let file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .write(true)
                .open(events_dir.join(fname.to_string()))
                .ok()
                .unwrap();

            let mut writer = BufWriter::new(file);
            let mut interval = time::interval(Duration::from_secs(5));

            loop {
                tokio::select! {
                    content = rx.recv() => {
                        if let Some(content) = content {
                            writeln!(&mut writer, "{}", content).unwrap();
                        } else {
                            break;
                        }
                    }
                    _ = interval.tick() => {
                        writer.flush().unwrap();
                    }
                }
            }
        });

        tx
    };
}

struct EventService;

#[derive(Serialize)]
struct Log<'a> {
    ts: u128,
    event: &'a Event<'a>,
}

impl EventLogger for EventService {
    fn log(&self, e: &Event) {
        let content = serdeconv::to_json_string(&Log {
            ts: timestamp(),
            event: e,
        })
        .unwrap();

        WRITER.send(content).unwrap();
    }
}

fn timestamp() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let start = SystemTime::now();
    start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_millis()
}

pub fn create_event_logger() -> impl EventLogger {
    EventService
}
