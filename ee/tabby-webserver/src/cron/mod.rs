mod db;
mod scheduler;

use std::sync::Arc;

use futures::Future;
use tokio::sync::broadcast::{self, error::RecvError, Receiver};
use tokio_cron_scheduler::{Job, JobScheduler};

use crate::schema::{auth::AuthenticationService, job::JobService, worker::WorkerService};

pub(crate) struct CronEvents {
    pub scheduler_job_succeeded: Receiver<()>,
}

pub trait StartListener<E> {
    fn start_listener<F, Fut>(&self, handler: F)
    where
        F: Fn(E) -> Fut + Send + 'static,
        Fut: Future + Send,
        E: Clone + Send + 'static;
}

impl<E> StartListener<E> for Receiver<E> {
    fn start_listener<F, Fut>(&self, handler: F)
    where
        F: Fn(E) -> Fut + Send + 'static,
        Fut: Future + Send,
        E: Clone + Send + 'static,
    {
        let mut recv = self.resubscribe();
        tokio::spawn(async move {
            loop {
                let event = match recv.recv().await {
                    Ok(event) => event,
                    Err(RecvError::Closed) => break,
                    Err(_) => continue,
                };
                handler(event).await;
            }
        });
    }
}

async fn new_job_scheduler(jobs: Vec<Job>) -> anyhow::Result<JobScheduler> {
    let scheduler = JobScheduler::new().await?;
    for job in jobs {
        scheduler.add(job).await?;
    }
    scheduler.start().await?;
    Ok(scheduler)
}

pub async fn run_cron(
    auth: Arc<dyn AuthenticationService>,
    job: Arc<dyn JobService>,
    worker: Arc<dyn WorkerService>,
    local_port: u16,
) -> CronEvents {
    let mut jobs = vec![];
    let (send_scheduler_complete, receive_scheduler_complete) = broadcast::channel::<()>(2);

    let job1 = db::refresh_token_job(auth.clone())
        .await
        .expect("failed to create refresh token cleanup job");
    jobs.push(job1);

    let job2 = db::password_reset_job(auth)
        .await
        .expect("failed to create password reset token cleanup job");
    jobs.push(job2);

    let job3 = scheduler::scheduler_job(job.clone(), worker, send_scheduler_complete, local_port)
        .await
        .expect("failed to create scheduler job");
    jobs.push(job3);

    let job4 = db::stale_job_runs_job(job)
        .await
        .expect("failed to create stale job runs cleanup job");
    jobs.push(job4);

    new_job_scheduler(jobs)
        .await
        .expect("failed to start job scheduler");
    CronEvents {
        scheduler_job_succeeded: receive_scheduler_complete,
    }
}
