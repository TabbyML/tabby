mod cron;
mod logger;

pub use cron::CronStream;
pub use logger::JobLoggerGuard;

pub trait Job {
    const NAME: &'static str;
}
