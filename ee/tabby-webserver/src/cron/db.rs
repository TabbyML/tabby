//! db maintenance jobs

use std::{sync::Arc, time::Duration};

use anyhow::Result;
use futures::Future;
use tokio_cron_scheduler::Job;
use tracing::error;

use crate::schema::{auth::AuthenticationService, job::JobService};

async fn service_job<F, S>(service: Arc<S>, job: fn(Arc<S>) -> F) -> Result<Job>
where
    F: Future<Output = Result<()>> + 'static + Send,
    S: Send + Sync + 'static + ?Sized,
{
    // job is run every 2 hours
    let job = Job::new_async("0 0 1/2 * * * *", move |_, _| {
        let auth = service.clone();
        Box::pin(async move {
            let res = job(auth.clone()).await;
            if let Err(e) = res {
                error!("failed to run cleanup job: {}", e);
            }
        })
    })?;

    Ok(job)
}

pub async fn refresh_token_job(auth: Arc<dyn AuthenticationService>) -> Result<Job> {
    service_job(auth, |auth| async move {
        Ok(auth.delete_expired_token().await?)
    })
    .await
}

pub async fn password_reset_job(auth: Arc<dyn AuthenticationService>) -> Result<Job> {
    service_job(auth, |auth| async move {
        Ok(auth.delete_expired_password_resets().await?)
    })
    .await
}

pub async fn stale_job_runs_job(jobs: Arc<dyn JobService>) -> Result<Job> {
    let job_res = Job::new_one_shot_async(Duration::from_secs(0), move |_, _| {
        let jobs = jobs.clone();
        Box::pin(async move {
            if let Err(e) = jobs.cleanup_stale_job_runs().await {
                error!("failed to cleanup stale job runs: {e}");
            }
        })
    });
    Ok(job_res?)
}
