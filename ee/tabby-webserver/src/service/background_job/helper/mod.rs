mod cron;
mod logger;

pub use cron::CronStream;
pub use logger::JobLogger;

pub trait Job: serde::Serialize {
    const NAME: &'static str;

    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn to_command(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}
