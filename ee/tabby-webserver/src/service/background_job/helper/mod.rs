mod cron;
mod logger;

pub use cron::CronStream;
pub use logger::JobLogger;
use tokio::sync::mpsc::error::SendError;

use super::BackgroundJobEvent;

pub trait Job {
    const NAME: &'static str;
}
