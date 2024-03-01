mod db;
mod scheduler;

use std::sync::Arc;

use futures::Future;
use tokio::sync::broadcast::{self, error::RecvError, Receiver};
use tokio_cron_scheduler::{Job, JobScheduler};
use tracing::error;

use crate::schema::{auth::AuthenticationService, job::JobService, worker::WorkerService};

pub(crate) struct CronEvents {
    pub scheduler_job_succeeded: Receiver<()>,
}

pub(crate) fn start_listener<E, F, Fut>(recv: &Receiver<E>, f: F)
where
    F: Fn(E) -> Fut + Send + 'static,
    Fut: Future + Send,
    E: Clone + Send + 'static,
{
    let mut recv = recv.resubscribe();
    tokio::spawn(async move {
        loop {
            let event = match recv.recv().await {
                Ok(event) => event,
                Err(RecvError::Closed) => break,
                Err(_) => continue,
            };
            f(event).await;
        }
    });
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
) -> Option<CronEvents> {
    let mut jobs = vec![];
    let (send_scheduler_complete, receive_scheduler_complete) = broadcast::channel::<()>(2);

    let Ok(job1) = db::refresh_token_job(auth.clone()).await else {
        error!("failed to create refresh token cleanup job");
        return None;
    };
    jobs.push(job1);

    let Ok(job2) = db::password_reset_job(auth).await else {
        error!("failed to create password reset token cleanup job");
        return None;
    };
    jobs.push(job2);

    let Ok(job3) =
        scheduler::scheduler_job(job.clone(), worker, send_scheduler_complete, local_port).await
    else {
        error!("failed to create scheduler job");
        return None;
    };
    jobs.push(job3);

    let Ok(job4) = db::stale_job_runs_job(job).await else {
        error!("failed to create stale job runs cleanup job");
        return None;
    };
    jobs.push(job4);

    if new_job_scheduler(jobs).await.is_err() {
        error!("failed to start job scheduler");
    };
    Some(CronEvents {
        scheduler_job_succeeded: receive_scheduler_complete,
    })
}
