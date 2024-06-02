mod cron;
mod logger;

pub use cron::CronStream;
pub use logger::JobLogger;
use tokio::sync::mpsc::error::SendError;

pub trait Job {
    const NAME: &'static str;
}

#[derive(Clone)]
pub struct JobQueue<T: Job> {
    sender: tokio::sync::mpsc::UnboundedSender<T>,
}

impl<T: Job> JobQueue<T> {
    pub fn new(sender: tokio::sync::mpsc::UnboundedSender<T>) -> Self {
        Self { sender }
    }

    pub fn enqueue(&self, job: T) -> Result<(), SendError<T>> {
        self.sender.send(job)
    }
}
