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
struct Event<'a, T>
where
    T: ?Sized + Serialize,
{
    name: &'a str,
    payload: &'a T,
}

pub fn log<T>(name: &str, payload: &T)
where
    T: ?Sized + Serialize,
{
    let mut writer = WRITER.lock().unwrap();

    serdeconv::to_json_writer(&Event { name, payload }, writer.by_ref()).unwrap();
    write!(writer, "\n").unwrap();
    writer.flush().unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Serialize)]
    struct MyString(String);

    #[test]
    fn it_works() {
        let content = MyString("abc".to_owned());
        log("abc", &content);
    }
}
