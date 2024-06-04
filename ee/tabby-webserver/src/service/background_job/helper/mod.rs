mod cron;
mod logger;

pub use cron::CronStream;
pub use logger::JobLogger;

pub trait Job {
    const NAME: &'static str;
}
