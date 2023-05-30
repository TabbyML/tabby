use chrono::Utc;
use lazy_static::lazy_static;
use serde::Serialize;
use std::fs;
use std::io::{BufWriter, Write};
use std::sync::Mutex;

lazy_static! {
    static ref WRITER: Mutex<BufWriter<fs::File>> = {
        let events_dir = &crate::path::EVENTS_DIR;
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

        Mutex::new(BufWriter::new(file))
    };
}

#[derive(Serialize)]
pub struct Choice<'a> {
    pub index: u32,
    pub text: &'a str,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Event<'a> {
    View {
        completion_id: &'a str,
        choice_index: u32,
    },
    Selected {
        completion_id: &'a str,
        choice_index: u32,
    },
    Completion {
        completion_id: &'a str,
        language: &'a str,
        prompt: &'a str,
        choices: Vec<Choice<'a>>,
    },
}

#[derive(Serialize)]
struct Log<'a> {
    ts: u128,
    event: &'a Event<'a>,
}

impl Event<'_> {
    pub fn log(&self) {
        let mut writer = WRITER.lock().unwrap();

        serdeconv::to_json_writer(
            &Log {
                ts: timestamp(),
                event: self,
            },
            writer.by_ref(),
        )
        .unwrap();
        write!(writer, "\n").unwrap();
        writer.flush().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        Event::View {
            completion_id: "abc".to_owned(),
            choice_index: 0,
        }
        .log();
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
