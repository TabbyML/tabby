use std::{
    fs,
    io::{BufWriter, Write},
    time::Duration,
};

use chrono::Utc;
use lazy_static::lazy_static;
use serde::Serialize;
use tokio::{
    sync::mpsc::{unbounded_channel, UnboundedSender},
    time::{self},
};

lazy_static! {
    static ref WRITER: UnboundedSender<String> = {
        let (tx, mut rx) = unbounded_channel::<String>();

        tokio::spawn(async move {
            let events_dir = crate::path::events_dir();
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

#[derive(Serialize)]
pub struct Choice<'a> {
    pub index: u32,
    pub text: &'a str,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectKind {
    Line,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Event<'a> {
    View {
        completion_id: &'a str,
        choice_index: u32,
    },
    Select {
        completion_id: &'a str,
        choice_index: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        kind: Option<SelectKind>,
    },
    Completion {
        completion_id: &'a str,
        language: &'a str,
        prompt: &'a str,
        segments: &'a Option<Segments>,
        choices: Vec<Choice<'a>>,
        user: Option<&'a str>,
    },
}
#[derive(Serialize)]
pub struct Segments {
    pub prefix: String,
    pub suffix: Option<String>,
}

#[derive(Serialize)]
struct Log<'a> {
    ts: u128,
    event: &'a Event<'a>,
}

impl Event<'_> {
    pub fn log(&self) {
        let content = serdeconv::to_json_string(&Log {
            ts: timestamp(),
            event: self,
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
